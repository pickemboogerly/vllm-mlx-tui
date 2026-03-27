import sys
from vllm_wave.tui.app import VLLMWaveApp

def main():
    app = VLLMWaveApp()
    app.run()

if __name__ == "__main__":
    main()
