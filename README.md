# OpenCV-Python 영상처리 실습 코드

교재 chap02 ~ chap12 실습 코드와, 모든 챕터가 공유하는 `Common` 모듈을 담고 있다.

## 디렉터리 구조

```
OpenCV-Python-ImageProcessing/
├── Common/              ← 전 챕터 공용 모듈 (저장소에 단 한 벌만 존재)
│   ├── fft2d.py  dft2d.py  dct2d.py  filters.py
│   └── histogram.py  hough.py  interpolation.py  knn.py  utils.py  calibration.py
├── chap02/ ... chap12/  ← 챕터별 실습 코드와 images/ 리소스
│   ├── chap11/header/   ← chap11 전용 보조 모듈 (paint_*, haar_*)
│   └── chap12/header/   ← chap12 전용 보조 모듈 (coin_*, plate_*)
├── .vscode/             ← 실행 경로 설정 (settings.json, launch.json)
└── pyproject.toml       ← Common 패키지 정의
```

> **`Common` 을 챕터 폴더로 복사하지 말 것.**
> 예전에 `chap09/Common`, `chap10/Common` 사본이 있었으나 제거했다.
> 사본을 만들면 수정이 한쪽에만 반영되어 디버깅이 매우 어려워진다.

## 최초 1회 설정

```powershell
# 저장소 루트에서
python -m pip install -e . --no-deps
```

이 명령은 `Common` 을 **현재 위치 그대로** 파이썬 환경에 등록한다(editable install).
파일을 복사하지 않으므로 `Common/` 을 수정하면 즉시 모든 챕터에 반영된다.
등록 후에는 작업 디렉터리나 터미널 종류와 무관하게 `from Common.fft2d import FFT` 가 동작한다.

필요한 패키지: `numpy`, `opencv-python`, `matplotlib` (이미 설치돼 있다면 위 명령으로 충분하다.)

## 실행 방법

### VS Code (권장)

파일을 열고 **Run** 버튼 또는 **F5**. 추가 설정은 필요 없다.
[.vscode/settings.json](./.vscode/settings.json) 과 [.vscode/launch.json](./.vscode/launch.json) 이
아래 두 가지를 자동으로 처리한다.

1. **작업 디렉터리를 그 파일의 폴더로 맞춘다** (`python.terminal.executeInFileDir`, `cwd: ${fileDirname}`).
   교재 코드가 `cv2.imread('images/mo2.jpg')` 처럼 **챕터 폴더 기준 상대 경로**를 쓰기 때문에 필수다.
2. **`PYTHONPATH` 에 저장소 루트와 두 `header` 폴더를 추가한다.**
   `Common` 과 `paint_init`·`coin_preprocess` 같은 챕터 보조 모듈을 찾게 해 준다.

### 터미널

작업 디렉터리를 **반드시 챕터 폴더로 옮긴 뒤** 실행한다.

```powershell
cd chap09
python 08.remove_moire.py
```

루트에서 `python chap09/08.remove_moire.py` 로 실행하면
`images/mo2.jpg` 를 찾지 못해 `영상 파일 읽기 에러` 가 난다.

chap11 · chap12 를 VS Code 통합 터미널 **밖에서** 실행할 때는 경로를 직접 추가해야 한다.
저장소 루트에서:

```powershell
$env:PYTHONPATH = "$PWD;$PWD\chap11\header;$PWD\chap12\header"
```

루트 경로가 들어가는 이유는 `chap03/09.module.py` 가
`import chap03.header_area` 처럼 **패키지 형식**으로 임포트하기 때문이다.

## 참고 사항

- **카메라 필요**: `chap04/17,18,19`, `chap10/10.warping_camera.py` 는 웹캠이 연결돼 있어야 한다.
- **`chap12/04.train_plate_data.py`**: 학습 영상 `images/plate/000.png ~ 069.png` 가 저장소에 없어
  실행되지 않는다. 다만 학습 결과물인 `SVMtrain.xml` 이 이미 있으므로
  `07.classify_plates.py` 등 이후 예제는 정상 동작한다.
- **NumPy 2.x 호환**: `np.complex`(chap09/08), `ndarray.itemset()`(chap06/01) 은
  NumPy 2.0 에서 제거되어 각각 `complex`, `arr[i, j] = v` 로 수정했다.

확인된 환경: Python 3.12 · NumPy 2.1 · OpenCV 4.12
