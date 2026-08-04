"""
主流程编排模块

提供一键运行整个 LLM 增强推荐 pipeline:
- Stage 1: 数据预处理
- Stage 2: 物品属性提取
- Stage 3: 用户画像构建
- Stage 4: Instruction 数据构建
- Stage 5: LLM 微调
- Stage 6: LLM 推理
- Stage 7: 评估

支持:
- 按阶段运行 (--stage)
- 断点续跑
- 配置覆盖
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import yaml

# 将 src 目录加入 path
sys.path.insert(0, str(Path(__file__).parent))

from data_utils import ensure_dirs, PROJECT_ROOT

logger = logging.getLogger(__name__)

# 各阶段定义及执行顺序
STAGES = [
    "preprocess",          # 数据预处理
    "extract_attributes",  # 物品属性提取
    "build_profiles",      # 用户画像构建
    "build_instructions",  # Instruction 数据构建
    "finetune",            # LLM 微调
    "inference",           # LLM 推理
    "evaluate",            # 评估
]


def load_config(config_path: Optional[str] = None) -> Dict:
    """加载 pipeline 配置"""
    if config_path is None:
        config_path = str(PROJECT_ROOT / "config" / "pipeline_config.yaml")

    config = {}
    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"已加载配置: {config_path}")
    else:
        logger.warning(f"配置文件不存在: {config_path}, 使用默认配置")

    return config


def run_stage(stage: str, config: Dict):
    """执行单个阶段"""
    logger.info("=" * 60)
    logger.info(f"执行阶段: {stage}")
    logger.info("=" * 60)

    start_time = time.time()

    if stage == "preprocess":
        from preprocess import preprocess
        preprocess_cfg = config.get("preprocess", {})
        data_cfg = config.get("data", {})
        preprocess(
            review_x_path=data_cfg.get("review_x_path"),
            review_y_path=data_cfg.get("review_y_path"),
            meta_x_path=data_cfg.get("meta_x_path"),
            meta_y_path=data_cfg.get("meta_y_path"),
            config=preprocess_cfg,
        )

    elif stage == "extract_attributes":
        attr_cfg = config.get("attribute_extraction", {})
        backend = attr_cfg.get("backend", "api")

        if backend == "convert":
            # 从已有 item_prompt 数据转换 (DeepSeek V4 结果)
            from convert_item_prompt import main as convert_main
            import sys as _sys
            dataset = attr_cfg.get("item_prompt_dataset", "GM")
            _sys.argv = ["convert_item_prompt.py", "--dataset", dataset]
            convert_main()

        elif backend == "api":
            from attribute_extraction import run_attribute_extraction
            if not attr_cfg.get("api_key"):
                logger.error(
                    "属性提取需要 API key，请在 config/pipeline_config.yaml 中配置 "
                    "attribute_extraction.api_key"
                )
                return
            run_attribute_extraction(attr_cfg)

        elif backend == "local":
            from attribute_extraction import run_attribute_extraction
            run_attribute_extraction(attr_cfg)

        else:
            logger.error(f"未知的属性提取后端: {backend}")

    elif stage == "build_profiles":
        from user_profile_builder import run_profile_building
        run_profile_building(config)

    elif stage == "build_instructions":
        from build_instruction_data import build_all_instruction_data
        build_all_instruction_data(config)

    elif stage == "finetune":
        from llm_finetune import train
        lora_cfg_path = PROJECT_ROOT / "config" / "lora_config.yaml"
        lora_cfg = {}
        if lora_cfg_path.exists():
            with open(lora_cfg_path) as f:
                lora_cfg = yaml.safe_load(f) or {}
        train(lora_cfg)

    elif stage == "inference":
        from llm_inference import run_inference
        lora_cfg_path = PROJECT_ROOT / "config" / "lora_config.yaml"
        lora_cfg = {}
        if lora_cfg_path.exists():
            with open(lora_cfg_path) as f:
                lora_cfg = yaml.safe_load(f) or {}
        run_inference(config=lora_cfg)

    elif stage == "evaluate":
        from evaluate import run_evaluation
        run_evaluation(config)

    else:
        logger.error(f"未知阶段: {stage}")
        return

    elapsed = time.time() - start_time
    logger.info(f"阶段 {stage} 完成, 耗时: {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="URLLM 增强推荐 Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
可用阶段 (按顺序):
  {chr(10).join(f'  {i+1}. {s}' for i, s in enumerate(STAGES))}

示例:
  # 运行全部阶段
  python run_pipeline.py

  # 只运行特定阶段
  python run_pipeline.py --stage build_profiles

  # 运行到某个阶段为止
  python run_pipeline.py --until build_instructions

  # 使用自定义配置
  python run_pipeline.py --config my_config.yaml
        """,
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=STAGES,
        help="只运行指定阶段",
    )
    parser.add_argument(
        "--until",
        type=str,
        choices=STAGES,
        help="运行到指定阶段为止 (含该阶段)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径 (默认: config/pipeline_config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要执行的阶段，不实际运行",
    )

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                str(PROJECT_ROOT / "outputs" / "pipeline.log"),
                encoding="utf-8",
                mode="a",
            ),
        ],
    )

    # 确保目录存在
    ensure_dirs()

    # 加载配置
    config = load_config(args.config)

    # 确定要执行的阶段
    if args.stage:
        stages_to_run = [args.stage]
    elif args.until:
        idx = STAGES.index(args.until)
        stages_to_run = STAGES[:idx + 1]
    else:
        stages_to_run = STAGES

    # 执行
    logger.info(f"Pipeline 启动: 将执行 {stages_to_run}")

    if args.dry_run:
        for s in stages_to_run:
            logger.info(f"  [DRY-RUN] {s}")
        return

    for stage in stages_to_run:
        try:
            run_stage(stage, config)
        except Exception as e:
            logger.error(f"阶段 {stage} 执行失败: {e}", exc_info=True)
            raise

    logger.info("Pipeline 全部完成！")


if __name__ == "__main__":
    main()
