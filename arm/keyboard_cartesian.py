"""末端键盘目标节点：先启动 run_executor.py，再运行本程序。"""
from pathlib import Path
import sys

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from keyboard_node import main
    sys.exit(main(mode='cartesian'))
