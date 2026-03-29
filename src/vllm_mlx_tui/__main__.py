import sys
from vllm_mlx_tui.tui.app import VLLMMlxTUIApp

def main():
    app = VLLMMlxTUIApp()
    app.run()

if __name__ == "__main__":
    main()
