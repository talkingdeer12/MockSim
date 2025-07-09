# MockSim

MockSim은 간단한 이벤트 기반 하드웨어 시뮬레이터로, 파이썬으로 작성되어 있습니다. PyTorch 모듈에 후크를 달아 신경망 레이어가 실행될 때 하드웨어 이벤트를 발생시키도록 구성되어 있습니다. 이를 통해 NPU, 메모리 서브시스템, 네트워크 온칩 등의 동작을 가볍게 모사할 수 있습니다.

## 주요 하드웨어 컴포넌트

각 컴포넌트는 실제 컴퓨터 아키텍처의 특징을 단순화하여 모델링합니다.

- **Engine** (`sim_core/engine.py`)
  - 전역 사이클 카운터를 관리하며 모든 `Event` 객체를 타임스탬프 순서로 처리합니다.
  - 이벤트 구동 방식으로 하드웨어 동작을 추상화합니다.

- **Router** (`sim_core/router.py`)
  - 4단계 파이프라인(RC→VA→SA→ST)과 가상 채널을 사용하는 2차원 메시 네트워크 라우터를 모델링합니다.
  - 실제 NoC(router)에서 사용되는 입력 버퍼, 교차바, 가상 채널 할당 등의 동작을 반영합니다.

- **Neural Processing Unit (NPU)** (`sim_hw/npu.py`)
  - 명령 파이프라인을 갖춘 단순한 연산 유닛을 모사합니다.
  - DMA를 통해 IOD(메모리)로부터 데이터를 읽고 쓰며, 파이프라인이 비어 있을 때 다음 명령을 시작합니다.

- **Control Processor (CP)** (`sim_hw/cp.py`)
  - 여러 NPU의 DMA 및 연산 단계를 순차/병렬로 제어합니다.
  - 프로그램별 스코어보드를 유지하고, `stream_id`를 이용해 동일 프로그램 내 서로 다른 작업 스트림을 겹쳐 실행할 수 있도록 합니다.
  - GPU의 커맨드 프로세서와 유사한 개념을 단순화한 형태입니다.

- **IOD** (`sim_hw/iod.py`)
  - 스택형 HBM 채널을 갖춘 메모리 컨트롤러를 모델링합니다.
  - 뱅크 그룹/뱅크/행 타이밍(tRP, tRCD, tCL)을 간략히 계산하여 메모리 접근 지연을 구합니다.

- **Event Logger** (`sim_core/logger.py`)
  - 각 모듈이 처리한 이벤트를 사이클 단위로 기록하여 Plotly 기반 타임라인을 생성합니다.

## 패키지 구성

- **`sim_core`** – 엔진, 이벤트 클래스, 라우터 및 공통 모듈 베이스 클래스를 포함합니다.
- **`sim_hw`** – CP, NPU, IOD 등 하드웨어 블록을 정의합니다.
- **`sim_ml`** – PyTorch 모듈과 후크가 위치하며, `llama3_decoder.py`와 `llama3_sim_hook.py`가 예제용으로 제공됩니다.

## 예제 실행 방법

1. PyTorch 설치(CPU 버전으로 충분합니다).
   ```bash
   pip install torch torchvision torchaudio
   ```
2. 샘플 스크립트 실행.
   ```bash
   python main.py
   ```
   스크립트는 간단한 메시를 구성하고 하드웨어 모듈을 등록한 뒤, 가짜 Llama3 디코더 블록을 실행합니다. 실행이 끝나면 `timeline.html`이 생성되며, 각 모듈의 파이프라인 활동을 주기별로 확인할 수 있습니다.

## 이벤트 추가하기

새로운 이벤트는 `Event` 객체를 생성하여 대상 모듈에 전달하면 됩니다.

```python
from sim_core.event import Event

# CP 모듈에 DMA 입력 명령을 하나 스케줄하는 예시
cp.send_event(Event(
    src=None,
    dst=cp,
    cycle=engine.current_cycle + 1,
    program="prog0",
    event_type="NPU_DMA_IN",
    payload={
        "program_cycles": 3,
        "in_size": 16,
        "out_size": 16,
        "dma_in_opcode_cycles": 2,
        "dma_out_opcode_cycles": 2,
        "cmd_opcode_cycles": 3,
        "stream_id": "A",   # 동시에 여러 작업을 돌리고 싶다면 서로 다른 stream_id 사용
        "eaddr": 0,
        "iaddr": 0,
    },
))
```

`cycle` 값은 이벤트가 실행되기를 원하는 시점을 지정합니다. `send_event`를 사용하면 목적지 버퍼가 가득 찬 경우 자동으로 재시도 이벤트가 생성됩니다.

## Stream ID로 작업 오버랩하기

CP는 하나의 프로그램 안에서 여러 스트림을 동시에 처리할 수 있도록 `stream_id` 필드를 사용합니다. 동일한 `stream_id`를 가진 명령들은 순서를 유지하지만, 서로 다른 `stream_id`는 독립적으로 진행됩니다. 이를 활용하면 타일 단위의 연산이나 여러 레이어를 파이프라인 형태로 겹쳐 실행할 수 있습니다.

예를 들어 `tests/test_tile_pipeline.py`에서는 각 타일을 다른 `stream_id`로 지정하여 DMA와 연산, 결과 쓰기를 겹쳐 수행하도록 구성되어 있습니다.

## 로깅과 타임라인 생성

시뮬레이터에서 이벤트 흐름을 시각화하고 싶다면 `EventLogger`를 사용합니다.

```python
engine = SimulatorEngine()
logger = EventLogger()
engine.set_logger(logger)
...
engine.run_until_idle()
logger.save_html("timeline.html")
```

`save_html` 호출 후 생성된 HTML 파일을 열면 사이클별 모듈 활동을 인터랙티브하게 살펴볼 수 있습니다.

## 테스트 실행

리포지터리에는 몇 가지 단위 테스트가 포함되어 있습니다.

```bash
python -m unittest discover tests
```

NPU 작업 흐름, 타일 파이프라인 및 무작위 트래픽 등 다양한 시나리오가 테스트에 포함되어 있습니다. 새 기능을 추가했다면 테스트가 모두 통과하는지 확인하세요.

