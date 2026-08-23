"""
主流程编排模块

提供一键运行整个 LLM 增强推荐 pipeline (通用版, 支持 AO/GM):
- Stage 1: 数据预处理 (解析 DG 产物)
- Stage 2: 物品属性提取 (复用 DG 预计算)
- Stage 3: 用户画像构建
- Stage 4: KNN 用户检索 (论文 §4.2.1)
- Stage 5: Instruction 数据构建 (含检索用户)
- Stage 6: LLM 微调
- Stage 7: LLM 推理
- Stage 8: Answer Refinement (论文 §4.2.3)
- Stage 9: 评估

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

from data_utils import ensure_dirs, set_dataset, get_output_dir, PROJECT_ROOT

logger = logging.getLogger(__name__)

# 各阶段定义及执行顺序
STAGES = [
    "preprocess",          # 1. 数据预处理
    "extract_attributes",  # 2. 物品属性提取
    "build_profiles",      # 3. 用户画像构建
    "retrieve_users",      # 4. KNN 用户检索 (论文 §4.2.1)
    "build_instructions",  # 5. Instruction 数据构建 (含检索用户)
    "finetune",            # 6. LLM 微调
    "inference",           # 7. LLM 推理
    "refine_answers",      # 8. 答案精炼: BM25+域检查+DG回退 (论文 §4.2.3)
    "evaluate",            # 9. 评估
]


def load_config(config_path: Optional[str] = None) -> Dict:
    """加载 pipeline 配置 (两数据集共用 config/pipeline_config.yaml)"""
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
        preprocess(config=preprocess_cfg)

    elif stage == "extract_attributes":
        from attribute_extraction import run_attribute_extraction
        attr_cfg = config.get("attribute_extraction", {})
        if not attr_cfg.get("use_precomputed") \
                and not attr_cfg.get("api_key") \
                and attr_cfg.get("backend", "api") == "api":
            logger.error(
                "属性提取需要 API key，请在 config/pipeline_config.yaml 中配置 "
                "attribute_extraction.api_key, "
                "或将 attribute_extraction.use_precomputed 置 true 使用 DG 预提取属性"
            )
            return
        run_attribute_extraction(attr_cfg)

    elif stage == "build_profiles":
        from user_profile_builder import run_profile_building
        run_profile_building(config)

    elif stage == "retrieve_users":
        from knn_retriever import run_user_retrieval
        run_user_retrieval(config)

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
        else:
            logger.warning(f"lora_config 不存在: {lora_cfg_path}, 使用默认参数")
        train(lora_cfg)

    elif stage == "inference":
        from llm_inference import run_inference
        lora_cfg_path = PROJECT_ROOT / "config" / "lora_config.yaml"
        lora_cfg = {}
        if lora_cfg_path.exists():
            with open(lora_cfg_path) as f:
                lora_cfg = yaml.safe_load(f) or {}
        else:
            logger.warning(f"lora_config 不存在: {lora_cfg_path}, 使用默认参数")
        run_inference(config=lora_cfg)

    elif stage == "refine_answers":
        from answer_refinement import refine_predictions
        refine_predictions(config)

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

    # 确保目录存在 (默认 AO, set_dataset 后会按新数据集再次创建子目录)
    ensure_dirs()

    # 加载配置
    config = load_config(args.config)

    # 设置数据集上下文 (AO / GM)
    ds_cfg = config.get("dataset", {})
    set_dataset(ds_cfg.get("name", "AO"), ds_cfg.get("dg_root"))

    # 数据集切换后, 确保对应子目录存在
    ensure_dirs()

    # 设置日志 (在 set_dataset 之后, pipeline.log 写入对应数据集目录)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                str(get_output_dir() / "pipeline.log"),
                encoding="utf-8",
                mode="a",
            ),
        ],
    )

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
