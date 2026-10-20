# 어라운드뷰 탑뷰 모아레 저감 — 구현 핸드오프 문서

**버전:** 1.0 (2026-09-30)
**대상 독자:** 사내 코드베이스를 분석하고 아키텍처를 설계할 코딩 에이전트, 그리고 검토할 담당 엔지니어
**전제:** 이 문서는 외부 세션(코드베이스 접근 없음)에서 리서치·시뮬레이션·설계 논의를 거쳐 확정한 전략을 담는다. 사내 코드에 대한 사실은 전부 "확인 필요"이며, 에이전트는 §7의 분석 체크리스트를 먼저 수행하고 그 결과를 바탕으로 §6의 지침에 따라 설계해야 한다.

---

## 0. 에이전트를 위한 요약과 작업 지시 (Agent brief)

**Problem.** The AVM top view (bird's-eye view) is rendered by warping fisheye camera frames on the GPU (OpenGL ES, per-vertex texture coordinates baked by the static calibration tool, camera frames bound as `samplerExternalOES` via EGLImage). Near the vehicle the ground is *minified* 4–12× (one output pixel covers 16–140 source texels), but the shader samples with a single bilinear tap. The result is resampling aliasing that shows as moiré on tiles, gratings, painted lines and fine road texture.

**Decision.** Fix it *before* sampling (pre-filter), never after. Chosen architecture:

1. Keep level 0 as the zero-copy external texture (unchanged).
2. Per camera per frame, build an **application-owned RGBA8 pyramid holding only 1/2, 1/4, 1/8 resolution** (3 fragment passes, 4-tap bilinear = 4×4 binomial kernel). No full-resolution copy.
3. In the static calibration/LUT tool, compute **per-triangle filter metadata** from the Jacobian of the output→source mapping: `lod`, major-axis vector, probe count `N`. Pass as `flat` attributes (or a metadata texture indexed by `gl_PrimitiveID`).
4. Top-view fragment shader: pick level (`external` for lod<1 mixed with pyramid level 0; `textureLod(pyramid)` for lod≥1) and average `N` probes along the major axis (bounded anisotropic filtering, Feline/EXT_texture_filter_anisotropic style, executed in software).
5. Interim measures deliverable earlier: request ISP edge-enhancement OFF for the top-view stream (tuning, not app code); adaptive supersampling on the external texture with tap spacing ≤ 2 texels, gated by the same metadata.

**Hard constraints.** Application-layer **OpenGL ES 3.2 + EGL only**. No chip-vendor SDK, no OpenVX, no ISP/scaler/IMR control APIs. ARM NEON is allowed but EGLImage gives no CPU pointer, so NEON is only for offline/calibration-time work unless a CPU-visible buffer contract is discovered. Camera zero-copy must remain.

**What the agent must do, in order.**
1. Run the codebase discovery checklist in §7 and fill the Runtime Capability Report template (Appendix C). Do not design before this is done.
2. Propose an architecture that follows §6 (component boundaries, data formats, frame sequencing, feature flags, A/B switch, instrumentation), citing the exact files/functions found in step 1.
3. Implement in the milestone order of §8 (M0→M5). Each milestone has exit criteria.
4. Never introduce vendor APIs, never post-filter the finished top view as the primary fix, never rely on mipmaps on `GL_TEXTURE_EXTERNAL_OES` (spec forbids it).

---

## 1. 배경과 문제 정의

### 1.1 현재 파이프라인 (알려진 사실)

- 4개(추정) 어안 카메라. 입력 해상도로 1280×944 와 1824×1536 두 종류가 확인됨.
- ISP 출력 DMA 버퍼를 EGLImage로 감싸 OpenGL ES 텍스처로 사용(제로카피). 텍스처 타깃은 `GL_TEXTURE_EXTERNAL_OES`로 추정(확인 필요, §7).
- 탑뷰는 메시 기반. 정적 공차보정(차량을 정위치에 두고 보정 타겟의 월드좌표–영상좌표 대응으로 외부 파라미터 계산)이 끝나면 탑뷰 메시의 버텍스마다 원영상 샘플링 위치(텍스처 좌표)를 계산해 굽는다. 런타임은 그 좌표로 원영상을 샘플해 탑뷰를 그린다.
- 프래그먼트 샘플링은 양선형 1탭(추정, 확인 필요).

### 1.2 증상과 원인

증상: 탑뷰의 차량 근접 구간에서 타일, 그레이팅, 미세 노면 텍스처, 주차선에 물결/줄무늬(모아레)가 보이고 주행 중 깜빡인다.

원인: **리샘플링 앨리어싱.** 어안 카메라는 차 옆 지면을 약 1.5~2 mm/px로 촘촘히 찍고, 탑뷰는 1~2 cm/px로 그린다. 출력 픽셀 하나의 원영상 풋프린트가 가로세로 4~12 텍셀인데 양선형은 2×2만 참조하므로, 나머지 텍셀의 고주파가 저주파로 접혀 들어온다(나이퀴스트). 샘플링 후에는 접힌 성분과 진짜 저주파를 구별할 수 없으므로 사후 필터(FFT 노치, 블러, 미디언)로는 복원되지 않는다.

### 1.3 정량 분석 (합성 모델, 사이드 카메라: 높이 0.95 m, 하향 60°, 190° 등거리 어안, 1824×1536)

| 카메라로부터 거리 | 1 cm/px: s_max / s_min | 2 cm/px: s_max / s_min | 이방성 | 해석 |
|---|---|---|---|---|
| 0.3 m | 5.6 / 5.3 | 11.1 / 10.5 | 1.1 | 등방 축소, 30~120 텍셀/픽셀 |
| 1.0 m | 4.0 / 2.7 | 8.1 / 5.4 | 1.5 | 강한 축소 |
| 2.0 m | 2.6 / 1.1 | 5.3 / 2.1 | 2.5 | 이방성 시작 |
| 3.0 m | 1.9 / 0.5 | 3.8 / 1.05 | 3.6 | 한 축 축소, 한 축 확대 |
| 5.0 m | 1.2 / 0.2 | 2.5 / 0.4 | 6.1 | 원거리: 해상도 부족이 문제 |

- s_max, s_min = 출력 1픽셀당 원영상 텍셀 수(자코비안 특이값). s>1 축소, s<1 확대.
- 면적 기준 1 cm/px 탑뷰의 84 %가 축소 구간(26 %가 2배 이상), 2 cm/px는 84 %가 2배 이상(26 %가 4배 이상).
- 전방 카메라(높이 0.65 m, 40°)는 근거리 s_max ≈ 8, 원거리 이방성 13~15.
- **고해상도 카메라(1824×1536)가 저해상도(1280×944)보다 모아레가 더 심하다**(축소율 1.4배).
- ISP 에지 강조/샤프닝은 원영상 나이퀴스트 근처 에너지를 키워 접힘을 키운다.

계산 스크립트: `minification_map.py`(축소율 지도), `moire_demo.py`(어안 합성 → 탑뷰 리샘플링 비교). 실제 카메라 파라미터로 바꿔 돌리면 사내 시스템의 지도가 나온다.

### 1.4 합성 실험 결과 (LF-PSNR: σ=2 px 저역 통과 후 PSNR, 접혀 들어온 저주파 거짓 무늬만 측정)

| 기법 | 1 cm 근거리 | 1 cm 중거리 | 2 cm 근거리 | 2 cm 중거리 | 평균 탭/픽셀 (2 cm) |
|---|---|---|---|---|---|
| 양선형 1탭 (현재) | 42.3 | 39.3 | 43.9 | 46.6 | 1 |
| 고정 4탭 SS (level 0) | 44.1 | 39.2 | 49.8 | 47.2 | 4 |
| 적응 SS ≤16탭 (level 0) | 44.5 | 38.3 | 50.7 | 47.1 | 11.5 |
| 등방 trilinear 밉맵 | 41.1 | 35.9 | 43.9 | 36.6 | 2 |
| **밉맵 + 이방성 프로브 (≤8)** | 44.1 | 39.3 | 49.3 | 47.4 | 3.2 프로브 × 2레벨 ≈ 6.5 |

결론: 등방 trilinear는 이방성 구간에서 과도 블러(최하점). 적응 SS는 품질은 좋으나 탭 수가 s²로 증가. 피라미드 + 프로브는 같은 품질을 상한이 고정된 비용으로 낸다.

---

## 2. 확정된 제약과 사양 근거

### 2.1 하드 제약

| 항목 | 내용 |
|---|---|
| API | 애플리케이션 계층 OpenGL ES 3.2 + EGL만 사용 |
| 금지 | 칩사 SDK, OpenVX, ISP/스케일러/IMR 제어 API, 벤더 전용 확장에 의존하는 설계 |
| 허용 | ARM NEON(단, EGLImage는 CPU 포인터를 제공하지 않음 → CPU 매핑 계약이 확인되기 전까지 NEON은 캘리브레이션 시점 계산에만 사용) |
| 유지 | 카메라 프레임 제로카피(외부 텍스처 경로 유지), 기존 프로듀서/컨슈머 acquire·release 계약 유지 |
| 선택 확장 | `GL_OES_EGL_image_external_essl3`(필수에 가까움), `GL_EXT_texture_filter_anisotropic`, `GL_EXT_disjoint_timer_query` — 있으면 사용, 없어도 동작해야 함 |

### 2.2 사양에서 확인된 사실 (Khronos 원문 기준)

- `OES_EGL_image_external`: "TEXTURE_EXTERNAL_OES targets never have more than a single LOD level." 최소 필터는 NEAREST/LINEAR만, 랩은 CLAMP_TO_EDGE만, `GenerateMipmap`은 INVALID_ENUM. → **외부 텍스처에는 밉맵이 없다.**
- `OES_EGL_image_external_essl3`: `samplerExternalOES`에 허용되는 함수는 `texture`, `textureProj`, `texelFetch`, `textureSize`뿐. `textureLod`, `textureGrad`, `textureOffset`, `textureGather`는 명시적으로 제외.
- `OES_EGL_image`: EGLImage 자매 텍스처를 respecify(예: `GenerateMipmap`)하면 구현이 새 저장소를 할당해 복사하고 EGLImage와의 관계를 끊는다(orphan + copy). → 외부 텍스처/EGLImage 2D 텍스처에서 밉맵을 만들려는 시도는 결국 풀 해상도 복사와 같다.
- `EXT_EGL_image_storage`: dma-buf EGLImage는 `GL_TEXTURE_2D` 또는 `EXTERNAL_OES`로만 import되고, 밉 체인은 EGLImage 소스에 있어야 하는데 dma-buf에는 밉 레벨 속성이 없다.
- `EXT_texture_filter_anisotropic`: 외부 텍스처에 설정하는 것이 오류는 아니지만 효과는 구현 정의. 밉맵이 없으면 최선의 경우 level 0에서 N개 양선형 탭. 신뢰하지 말고 실측만.
- Arm GPU Best Practices: trilinear 2×, N× AF는 N×(8× trilinear AF = 16×), "2× bilinear AF가 등방 trilinear보다 빠르고 좋다", `textureGrad`는 `texture`/`textureLod`보다 느림, highp 샘플러는 절반 속도, 밉맵 없는 산발 샘플은 텍스처 캐시 실패의 대표 원인.
- OpenGL ES 3.2 §8.14.1: LOD λ = log2(ρ), ρ ≈ 화면 미분의 최대 열 길이 근사 허용 → 전단(shear)에서 풋프린트 과소평가 가능. `EXT_texture_filter_anisotropic`: N = min(⌈Pmax/Pmin⌉, maxAniso), λ = log2(Pmax/N).

### 2.3 아직 모르는 것 (§7에서 확인)

SoC/GPU/드라이버, 실제 확장 목록, EGLImage 타깃과 포맷(FOURCC, 플레인, 스트라이드, 크로마 위치, 범위), 버퍼 소유권/펜스, 출력 해상도와 FPS, 카메라 수, 메시 구조(버텍스 간격, 삼각형 토폴로지, 인덱스 여부), 셰이더 구조(GLSL 버전, precision, 블렌딩 방식), 3D 보울 뷰 유무.

---

## 3. 전략과 결정

### 3.1 채택안: "레벨 0 외부 텍스처 + 1/2 이하 GPU 소유 피라미드 + 정적 삼각형별 필터 메타데이터 + 장축 프로브"

핵심 아이디어 세 가지.

1. **프리필터.** 출력 픽셀의 풋프린트(자코비안)에 맞춰 원영상을 샘플하기 전에 평균한다. 넓은 등방 축소는 피라미드(밉맵)가, 길쭉한 이방성 풋프린트는 장축 프로브가 담당한다.
2. **원본은 복사하지 않는다.** 외부 텍스처를 레벨 0으로 그대로 쓰고 1/2·1/4·1/8만 소유 RGBA8 텍스처로 만든다. 저장·트래픽이 "풀 해상도 복사 + 전체 밉체인"(≈4P/3 텍셀)의 1/4(≈P/3). 1824×1536 4ch 기준 소유 피라미드 ≈14 MiB, 명목 트래픽 30 fps ≈1.4 GB/s(풀 체인 ≈4.1 GB/s).
3. **자코비안은 정적이다.** 탑뷰 매핑은 공차보정 때만 바뀌므로 LOD·장축·프로브 수를 실시간 계산할 이유가 없다. 캘리브레이션 툴이 텍스처 좌표를 굽는 자리에서 삼각형별 메타데이터를 함께 굽는다.

### 3.2 단계 전략

| 단계 | 내용 | 기대 효과 | 비용 |
|---|---|---|---|
| 0 | 탑뷰용 스트림 ISP 에지 강조/샤프닝 OFF 요청(ISP 튜닝, 앱 코드 아님). 원인 단계 분리(카메라 출력/워프/심/디스플레이 ROI 덤프) | 근거리 모아레 감소, 원인 확정 | 0 |
| 1 | 적응 슈퍼샘플링(외부 텍스처 level 0, 양선형 탭 간격 ≤2 텍셀, N=⌈s_max/2⌉ ≤ 8) + 삼각형별 메타데이터 도입 | 이방성 구간·중거리 개선. 1 cm/px 근거리 1.5 m 이내, 2 cm/px 3 m 이내는 부족 | 평균 5~11탭/픽셀 |
| 2 | 소유 피라미드(1/2~1/8) 생성 + 레벨 선택 + 장축 프로브 | 본 해법. 근거리 포함 전 구간 | 3 패스 + 평균 6~7탭/픽셀 |
| 3 | 튜닝(LOD bias, N_max), 피라미드 레벨에서 HW 이방성 A/B, 시간축 프로브 교대 | 미세 조정 | — |

### 3.3 기각한 대안과 이유 (결정 기록)

| 대안 | 기각 이유 |
|---|---|
| 완성된 탑뷰 사후 필터(FFT 노치, 블러, 미디언) | 앨리어싱은 비가역. 무늬 주파수가 위치·주행에 따라 변함. 1080p FFT/프레임 비용 과다 |
| MSAA | 프래그먼트 셰이더는 픽셀당 1회 → 텍스처 앨리어싱에 무관 |
| SSAA 2×2 | 2배 축소까지만 커버, 8~12배엔 무력. 4× 셰이딩 |
| 풀 해상도 복사 + `glGenerateMipmap` + HW trilinear/AF | 동작은 하지만 트래픽 4배. 필터가 구현 정의(대개 2×2 박스) |
| 정확 EWA(셰이더 직접 컨볼루션) | 비용이 풋프린트 면적 비례. 데스크톱에서도 HW AF의 5.5 % 속도. 오프라인 기준 영상 생성에만 사용 |
| Summed-area table | 32비트 테이블(~45 MB), 프레임당 생성 패스, 축 정렬 박스만 |
| 등방 trilinear만 | 이방성 구간 과도 블러(실험 최하점) |
| TAA | 프리필터 대체 불가. 입체물 고스트. 나중에 프로브 세트 교대 정도만 |
| 딥러닝 demoiré | 화면 격자 모아레용, 1080p 48 ms(플래그십 폰), 정보 손실을 환각으로 메움 |
| 컴퓨트 셰이더 피라미드 | 타일 기반 모바일 GPU에서 프래그먼트 패스보다 빠르다는 보장 없음. `r8/rg8`은 ES 3.2 코어 이미지 포맷 아님. 실험 후보로만 |
| NEON 피라미드 | EGLImage CPU 포인터 없음 → readback→CPU→업로드 경로가 되어 비권장. CPU 매핑 계약이 확인되면 재검토 |
| ISP/HW 스케일러가 피라미드 생성 | 제약(벤더 API 금지)으로 불가. 제약 해제 시 GPU 생성 패스만 제거되는 변형 |

### 3.4 IP 주의

서라운드뷰에서 "픽셀 밀도(축소율)로 안티에일리어싱 탭 수를 가변 적용"하는 방식은 Valeo/Connaught 특허(US 2021/0398256 "Enhanced anti-aliasing filtering on a GPU", 관련 US 11302062)가 존재한다. 1단계(적응 SS)의 밀도 게이팅은 이 청구항과 유사할 수 있으므로 **양산 적용 전 사내 IP 검토**를 요청한다. 반면 밉맵/trilinear(Williams 1983), EWA(Greene & Heckbert 1986), 장축 프로브(Feline, McCormack 1999; EXT_texture_filter_anisotropic)와 정적 LOD 사전계산은 오래된 공지 기술로 리스크가 낮다. 2단계 본 해법이 안전한 쪽이다.

---

## 4. 알고리즘 사양

### 4.1 삼각형별 자코비안과 특이값

탑뷰 메시의 삼각형 꼭짓점 출력 좌표 p0,p1,p2(출력 픽셀 단위), 대응 원영상 좌표 q0,q1,q2(원영상 픽셀 단위; 정규화 UV라면 (W,H)를 곱해 픽셀로 환산).

```
P = [p1-p0, p2-p0]   (2x2, 열벡터)
Q = [q1-q0, q2-q0]
J = Q · inv(P)       # 단위: 원영상 px / 출력 px.  J = [[a, b], [c, d]]
```

2×2 SVD(라이브러리 없이):

```
A = a² + b²;  B = a·c + b·d;  C = c² + d²        # G = J·Jᵀ
L = (A + C + sqrt((A - C)² + 4B²)) / 2
s_max = sqrt(L)
s_min = |a·d - b·c| / s_max                       # s_max > 0
theta = 0.5 · atan2(2B, A - C)                    # 원영상 공간 장축 각도
major_px = s_max · (cos theta, sin theta)         # 원영상 px 단위 장축 벡터
```

주의:
- 이웃 버텍스 UV 차이를 쓸 때 반드시 출력 픽셀 간격으로 나눈다(16 px 떨어진 버텍스의 UV 차이 96 px → J 성분 6).
- `dFdx/dFdy` 열 길이 비교(하드웨어식 근사)는 전단에서 과소평가한다(J=[[8,8],[0,1]]: 열 길이 8.0/8.06, 특이값 11.3/0.71). 특이값을 쓴다.
- 선형 보간 UV에서는 J가 삼각형마다 상수. 한 버텍스는 J가 다른 여러 삼각형에 속하므로 메타데이터는 삼각형 단위가 정확하다.
- 3D 보울 뷰처럼 가상 카메라가 움직이는 뷰는 J가 정적이 아니다. 그 뷰는 프래그먼트에서 `dFdx/dFdy(uv·texSize)`로 같은 공식을 실시간 계산한다(비용 증가, 같은 필터 계획 함수 재사용). 이 문서는 고정 탑뷰를 대상으로 한다.

#### 4.1.1 제품 메시가 8픽셀 격자일 때 (확인된 사실: 버텍스 상하좌우 8 px 간격, 중간은 래스터라이저 보간)

J는 "비율"(미분)이므로 버텍스 간격이 1 px이든 8 px이든 정의는 같다. 격자 간격 g = 8 px에서는 이웃 버텍스의 원영상 좌표 차이를 g로 나누면 그것이 곧 J의 열이다. 추가 투영 계산이 필요 없고, 툴이 이미 갖고 있는 버텍스별 (u,v)만으로 된다.

격자 셀의 네 꼭짓점 원영상 좌표를 q00(좌하), q10(우하), q01(좌상), q11(우상)이라 하고(픽셀 단위), 셀이 두 삼각형 A = (q00, q10, q01), B = (q11, q01, q10)로 나뉜다면:

```
J_A = [ (q10 - q00) / g ,  (q01 - q00) / g ]      # 열 1 = 출력 x로 1 px 이동, 열 2 = 출력 y로 1 px 이동
J_B = [ (q11 - q01) / g ,  (q11 - q10) / g ]
```

이것이 §4.1의 J = Q·inv(P)에서 P = [[g,0],[0,g]]인 특수한 경우다. 래스터라이저가 UV를 삼각형 안에서 선형 보간하므로(직교 탑뷰, w=1), 프래그먼트가 실제로 겪는 미분값은 정확히 J_A, J_B로 삼각형 안에서 상수이고, 프래그먼트 셰이더에서 `dFdx(vUV)·texSize`를 찍어 보면 같은 값이 나와야 한다(검증 포인트). 삼각형의 정점 순서(대각선 방향)는 메시 생성 코드에서 확인해 그대로 맞춘다.

셀 안에서 J_A와 J_B는 조금 다르다(매핑이 셀 안에서 정확히 어파인이 아니므로). 근거리(0.3~0.5 m)에서도 s_max가 20 px에 걸쳐 5.6→5.1 정도로 변하므로 8 px 셀 안의 차이는 수 % 수준이며, 삼각형별로 정확히 넣으면 그 차이까지 반영되고, 버텍스별(방식 C)로 넣고 보간해도 오차는 작다. 버텍스별로 넣을 때는 그 버텍스를 공유하는 삼각형들의 J로 각각 s_max, s_min, 장축 벡터를 구해 평균한 뒤 **s 값과 장축 벡터를 보간**하고(lod는 로그라 직접 보간하지 않음), 프래그먼트에서 m, lod, N을 계산한다.

격자 메시 참조 코드(버텍스 배열 `q[row, col, 2]`, 픽셀 단위, 행이 출력 y):

```python
def grid_meta(q, g, W, H, n_max=8, bias=0.3):
    """q: (R, C, 2) 원영상 px 좌표, g: 버텍스 간격(출력 px). 셀 (r,c)의 두 삼각형 메타를 반환."""
    out = {}
    for r in range(q.shape[0] - 1):
        for c in range(q.shape[1] - 1):
            q00, q10, q01, q11 = q[r, c], q[r, c+1], q[r+1, c], q[r+1, c+1]
            JA = np.stack([(q10 - q00) / g, (q01 - q00) / g], axis=1)
            JB = np.stack([(q11 - q01) / g, (q11 - q10) / g], axis=1)
            out[(r, c, 'A')] = meta_from_J(JA, W, H, n_max, bias)   # Appendix B 의 SVD·필터 계획 부분
            out[(r, c, 'B')] = meta_from_J(JB, W, H, n_max, bias)
    return out
```

메시가 픽셀이 아니라 NDC/클립 좌표로 저장되어 있으면 g를 뷰포트 크기로 픽셀 환산한다(예: NDC 간격 Δ → g = Δ·뷰포트폭/2). 뷰(줌 레벨)마다 g의 픽셀 환산이 달라지므로 뷰별 메타 세트가 필요하다.

### 4.2 필터 계획 (bounded anisotropic)

```
N_max = 8                                   # 초기 4~8 스윕
m   = max(1, s_min, s_max / N_max)          # 등방 프리필터 폭 (원영상 px) = 프로브 하나가 덮는 폭
lod = log2(m) + bias                        # 원본 기준 LOD. bias 초기값 +0.3 (앨리어싱보다 블러를 택함)
N   = clamp(ceil(s_max / m - 1e-6), 1, N_max)   # 장축 프로브 수. 간격 s_max/N ≤ m → 해당 레벨 텍셀 1 이하
```

예: 6×1 → m=1, lod=0, N=6. 6×6 → m=6, lod=2.58, N=1. 32×1(cap 8) → m=4, lod=2, N=8(단축이 4배 흐려지는 것을 감수).

저장 형식(권장): `vec4 meta = (lod, major_px.x / W_src, major_px.y / H_src, N)` — 정규화 UV 단위의 장축 벡터. 확대 구간(s_max ≤ 1)은 (0, 0, 0, 1).

### 4.3 피라미드 생성

- 카메라당 텍스처 1개: `glTexStorage2D(GL_TEXTURE_2D, 3, GL_RGBA8, W/2, H/2)` → 레벨 0 = 원본 1/2, 1 = 1/4, 2 = 1/8. (1824×1536 → 912×768, 456×384, 228×192.)
- 샘플러 상태: `GL_LINEAR_MIPMAP_LINEAR` / `GL_LINEAR`, `GL_CLAMP_TO_EDGE`, `TEXTURE_MAX_LEVEL = 2`. 셰이더는 `mediump sampler2D`.
- 패스 0: 외부 텍스처 → 레벨 0. 패스 1: 레벨 0 → 레벨 1. 패스 2: 레벨 1 → 레벨 2. 각 패스는 풀스크린 삼각형, FBO 컬러 어태치먼트 = 목적 레벨(`glFramebufferTexture2D(..., tex, level)`).
- 커널: 목적지 텍셀 중심 c에서 소스 텍셀 단위 (±0.75, ±0.75)의 양선형 4탭 평균. 정렬된 2:1 축소에서 이것은 4×4 이항 커널 [1,3,3,1]²/64와 수치적으로 동일(검증 오차 < 1e-13). 새 나이퀴스트에서 −9 dB(2×2 박스는 −3 dB).
- 피드백 루프 회피: 레벨 i를 그리는 동안 같은 텍스처의 레벨 i−1을 샘플하므로 `TEXTURE_BASE_LEVEL = TEXTURE_MAX_LEVEL = i−1`로 묶는다(ES 3.x feedback 규칙: 어태치된 레벨이 샘플 가능한 레벨 범위 밖이면 정의된 동작). 생성 후 BASE=0, MAX=2로 복원. 드라이버 이슈가 있으면 레벨별 별도 텍스처 + 수동 trilinear로 대체(디버그 기준 구현으로도 유용).
- 동기화: 같은 컨텍스트에서 FBO 렌더 → 이후 `sampler2D` 페치는 별도 `glMemoryBarrier` 불필요. 카메라 버퍼 acquire/release 계약은 기존 것을 유지(피라미드 패스가 외부 텍스처를 읽는 동안 버퍼가 유효해야 함).
- `glGenerateMipmap`은 쓰지 않는다(필터 구현 정의).

### 4.4 탑뷰 프래그먼트 샘플링

```glsl
#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision mediump float;

uniform samplerExternalOES uCam;   // 레벨 0: 원본 (제로카피)
uniform sampler2D uPyr;            // 소유 피라미드: level0 = 1/2, level1 = 1/4, level2 = 1/8
in vec2 vUV;                       // 정규화 텍스처 좌표 (모든 레벨 공통)
flat in vec4 vMeta;                // x: lod(원본 기준), yz: 장축 벡터(정규화 uv), w: N
out vec4 fragColor;

vec4 readAt(vec2 uv, float lod) {
    if (lod <= 0.0) return texture(uCam, uv);                                  // 확대 구간: 현재와 동일
    if (lod <  1.0) return mix(texture(uCam, uv), textureLod(uPyr, uv, 0.0), lod);
    return textureLod(uPyr, uv, lod - 1.0);                                    // 피라미드 내부 HW trilinear
}

void main() {
    float N = vMeta.w;
    vec4 acc = vec4(0.0);
    for (int i = 0; i < 8; ++i) {
        if (float(i) >= N) break;
        float t = (float(i) + 0.5) / N - 0.5;                                  // [-0.5, 0.5) 등간격
        acc += readAt(vUV + t * vMeta.yz, vMeta.x);
    }
    fragColor = acc / N;   // 이후 기존 블렌딩(심 가중치 등) 적용
}
```

구현 주의:
- 프로브 가중치는 균등(박스). 가우시안 가중이 필요하면 `w_i = exp(-2·t_i²)` 정규화로 교체(품질 차이 미미, 우선 균등).
- `N`은 삼각형별 상수이므로 2×2 쿼드 내 분기 발산은 삼각형 경계에서만 발생. 메시 영역별로 N을 계단식으로 굽는 것이 더 안전.
- 외부 텍스처의 YUV→RGB 변환과 필터링 순서는 사양상 미정의. 피라미드는 외부 텍스처의 변환된 RGB를 읽어 만들므로 레벨 간 색은 일치해야 하지만, 실기에서 평탄한 회색 카드로 레벨 0/피라미드 경계(lod≈1 구간)에 색 단차가 없는지 확인한다.
- 기존 셰이더가 블렌딩 가중치, 감마, 색 보정 등을 하고 있다면 그 뒤에 그대로 붙인다. 필터링은 반드시 블렌딩 **전**에, 두 카메라 모두 같은 방식으로.

### 4.5 1단계 적응 슈퍼샘플링 (외부 텍스처만)

```glsl
// vMeta: x 미사용, yz 장축 벡터(정규화 uv), w = N_ss = clamp(ceil(s_max / 2), 1, 8)
vec4 acc = vec4(0.0);
for (int i = 0; i < 8; ++i) {
    if (float(i) >= N) break;
    float t = (float(i) + 0.5) / N - 0.5;
    acc += texture(uCam, vUV + t * vMeta.yz);   // 탭 간격 = s_max/N ≤ 2 텍셀 → 끊김 없는 박스
}
```

규칙: 양선형 한 탭이 2텍셀을 덮으므로 간격을 2 이하로. 간격을 넓혀 탭을 아끼면 빈틈의 무늬가 다시 접힌다. 단축 s_min > 2인 구간(근거리)은 이 방식으로 부족 → 2단계 필요. 선택: 픽셀별 고정 지터(±0.25 텍셀)로 잔여 앨리어싱을 노이즈화.

### 4.6 메타데이터 전달 방식 (메시 구조 분석 후 선택)

| 방식 | 조건 | 장점 | 단점 |
|---|---|---|---|
| A. 비인덱스 스트림 + `flat` 속성 | 메시를 삼각형당 3버텍스로 펼칠 수 있을 때 | 정확, 단순 | 버텍스 수 증가(위치/UV는 기존 인덱스 버퍼 유지하고 메타 스트림만 별도 VBO로 두는 변형 가능하나 GL ES에는 프리미티브 속성이 없어 결국 펼쳐야 함) |
| B. 메타데이터 텍스처 + `gl_PrimitiveID` | ES 3.2(`gl_PrimitiveID`는 ES 3.2 프래그먼트에서 사용 가능) | 버텍스 버퍼 무변경. 삼각형 인덱스 = 드로우 내 프리미티브 순번 | `texelFetch` 1회 추가, 드로우 분할 시 오프셋 관리 |
| C. 버텍스별 속성 + 부드러운 보간 | 메시가 조밀(버텍스 간격 ≤ 16 출력 px)하고 J 변화가 완만할 때 | 변경 최소 | 근사. lod(로그)를 직접 보간하지 말고 s_max, s_min, 장축을 보간한 뒤 프래그먼트에서 m, lod, N 계산 |

권장: 메시 분석 결과 인덱스 메시이면 B, 이미 비인덱스이면 A, 빠른 프로토타입은 C.

---

## 5. 성능 모델과 예산

| 출력 | 픽셀/s | 채택 경로 (≈6.5탭) | 적응 SS 2 cm (≈11.5탭, level 0) |
|---|---|---|---|
| 1280×720 @30 | 27.6 M | 0.18 G탭/s | 0.32 G탭/s |
| 1920×1080 @30 | 62.2 M | 0.40 G탭/s | 0.72 G탭/s |
| 1920×1080 @60 | 124 M | 0.81 G탭/s | 1.43 G탭/s |

- 임베디드 GPU 텍스처 처리량 참고: PowerVR GE8430(TDA4VM) 추정 3~6 GTex/s, Mali-G78 코어당 4텍셀/clk, Vivante GC7000L 1.6 GTex/s. 피라미드 경로는 레벨 텍셀이 픽셀 크기와 맞아 캐시 적중이 높고, level 0 산발 샘플(현재·적응 SS)은 캐시에 불리하다.
- 피라미드 생성: 카메라당 프레임당 원본 읽기 1회 + 소유 피라미드 쓰기/읽기. 1824×1536 4ch 30 fps 명목 ≈1.4 GB/s. 산술 모델이므로 실측으로 대체.
- 측정: `GL_EXT_disjoint_timer_query`(있을 때; `GL_GPU_DISJOINT_EXT` 확인), 없으면 `glFenceSync` + CPU 타임스탬프(종단 지연, 순수 GPU 시간 아님). 벤더 프로파일러 사용은 허용(앱 코드 의존 아님).
- 합격 예산(초안): 피라미드 3패스 + 탑뷰 드로우 추가분 ≤ 기존 프레임 GPU 시간의 30 %. 실측 후 조정.

---

## 6. 아키텍처 지침 (설계 시 준수)

### 6.1 컴포넌트 경계

1. **CalibrationTool / LUT 생성기 확장** — 기존 "월드 → 영상 좌표" 매핑 함수를 재사용해 삼각형별 J → 메타데이터 계산. 출력 포맷에 버전 필드를 넣고 런타임이 구버전(메타 없음)도 읽을 수 있게 한다. 파라미터: gsd(출력 m/px), 출력 해상도, N_max, bias. NEON/파이썬 어느 쪽이든 무방(오프라인).
2. **PyramidGenerator (런타임, 카메라당 1 인스턴스)** — 리소스: RGBA8 3레벨 텍스처, FBO 3개(또는 1개 재어태치), 다운샘플 프로그램 1개(외부 텍스처 입력용/ sampler2D 입력용 두 변형), 풀스크린 VAO. API: `init(W,H)`, `build(externalTex)`, `texture()`. 상태 복원(BASE/MAX_LEVEL) 책임을 내부에 둔다.
3. **TopViewRenderer 변경** — 셰이더 변형을 런타임 플래그로 선택: `MODE_BILINEAR`(현재, 회귀 기준), `MODE_ADAPTIVE_SS`(1단계), `MODE_PYRAMID_PROBES`(2단계). 새 입력: 메타데이터(속성 또는 텍스처), `uPyr`. 블렌딩·심 로직은 건드리지 않는다.
4. **Config / Tuning** — N_max, bias, 모드, 디버그 뷰(레벨 시각화: lod를 색으로), 카메라별 on/off.
5. **Instrumentation** — 단계별 ROI 덤프(카메라 출력, 탑뷰, 최종), GPU 타이머, 프레임 드롭 카운터.

### 6.2 프레임 순서

```
카메라 프레임 acquire (기존)
  → [새] PyramidGenerator.build(cam_i) × 4   (외부 텍스처 읽기; 버퍼 유효 구간 안에서)
  → 탑뷰 드로우 (기존 메시 + 새 속성/유니폼)
  → 블렌딩/합성 (기존)
  → 프레임 release (기존)
```

- 피라미드는 탑뷰 드로우 직전, 같은 GL 컨텍스트에서 만든다. 멀티스레드/멀티컨텍스트 구조라면 기존 동기화 원칙(펜스)을 따르고, 새 배리어를 임의로 추가하지 않는다.
- 카메라 프레임이 갱신되지 않은 프레임(프레임 반복)에는 피라미드를 재생성하지 않는다(프레임 시퀀스 번호로 판단).

### 6.3 데이터 형식

- 메타데이터: 삼각형당 `float4 (lod, major_u, major_v, N)`; 파일에는 s_max, s_min, theta도 함께 저장해 런타임에서 N_max/bias를 바꿔 재계산할 수 있게 한다.
- 뷰(줌 레벨, gsd)마다 별도 메타데이터 세트. 3D 보울 뷰는 런타임 미분 경로.

### 6.4 호환성과 안전

- 기존 경로(`MODE_BILINEAR`)는 삭제하지 않고 런타임 A/B 스위치로 유지.
- 확장 부재 시 폴백: essl3 외부 샘플러가 없으면(ESSL 1.00만) `texture2D` 기반 변형; 이 경우 `textureLod`가 없으므로 피라미드는 레벨별 별도 텍스처 + 수동 mix.
- 메타데이터 파일 부재/버전 불일치 시 자동으로 `MODE_BILINEAR`.
- 리소스 실패(FBO incomplete 등) 시 로그 후 `MODE_BILINEAR`.

### 6.5 리스크

| 리스크 | 완화 |
|---|---|
| 외부 텍스처 색 변환과 피라미드 색 불일치 | 회색 카드 테스트, lod≈1 경계 확인, 필요 시 피라미드 생성 시 동일 셰이더 경로 사용 |
| 드라이버가 같은 텍스처 레벨 간 렌더/샘플을 잘못 처리 | 레벨별 별도 텍스처 폴백 |
| 프레임 시간 초과 | N_max 낮춤(4), 피라미드 레벨 수 2로, 720p 모드 |
| 삼각형 경계에서 필터 폭 불연속이 보임 | bias 상향, 메시 조밀화, 방식 C(보간) 검토 |
| 주행 중 잔여 깜빡임 | 프레임별 프로브 오프셋 교대(0, ±1/4) |
| IP | §3.4 검토 |

---

## 7. 코드베이스 분석 체크리스트 (설계 전에 반드시 수행)

에이전트는 아래를 조사해 Appendix C 템플릿을 채운다. 각 항목에 **파일 경로와 함수/심볼**을 명시한다.

### 7.1 카메라 입력과 EGLImage
- EGLImage 생성 코드: `eglCreateImageKHR` 호출 위치, 타깃(`EGL_LINUX_DMA_BUF_EXT`? 플랫폼 네이티브 버퍼?), 속성(FOURCC, 플레인 수, 스트라이드, 오프셋, 색공간/범위/크로마 위치 속성).
- GL 바인딩: `glEGLImageTargetTexture2DOES`의 타깃이 `GL_TEXTURE_EXTERNAL_OES`인지 `GL_TEXTURE_2D`인지. 텍스처 파라미터 설정(min/mag 필터, 랩).
- 프레임 소유권: acquire/release 시점, 펜스(`EGL_KHR_fence_sync`, `EGL_ANDROID_native_fence_sync`, V4L2 dqbuf/qbuf 등), 버퍼 풀 크기, 프레임 시퀀스 번호 유무.
- CPU 매핑: 버퍼를 CPU가 map하는 코드가 있는지(있다면 NEON 경로 재검토 근거).

### 7.2 메시와 캘리브레이션 산출물
- 탑뷰 메시 생성 코드: 버텍스 포맷(위치, UV, 블렌딩 가중치, 카메라 ID 등), 인덱스 버퍼 유무, 삼각형 토폴로지(스트립/리스트), 버텍스 간격(출력 px), 카메라별 메시인지 통합 메시인지.
- UV 계산 경로: 월드(X,Y,0) → 영상(u,v) 함수(어안 모델, 내부/외부 파라미터), 정규화 여부, 크롭/스케일 적용 여부.
- 산출물 파일 포맷과 로더: 버전 필드, 뷰별 세트, 갱신 시점(공차보정 완료 시).
- 심/블렌딩: 가중치가 버텍스 속성인지 텍스처인지, 블렌딩이 같은 패스에서 이뤄지는지.

### 7.3 셰이더와 렌더 루프
- 탑뷰 버텍스/프래그먼트 셰이더 소스: GLSL 버전, `samplerExternalOES` 사용 여부와 함수, precision, 블렌딩/감마/색 보정 위치.
- 렌더 루프: 패스 구성, FBO 사용 현황(재사용 가능한 오프스크린 패스), 출력 해상도, 목표 FPS, 멀티컨텍스트/스레드 구조, 기존 타이머/프로파일 코드.
- 3D 보울 뷰 등 가상 카메라가 움직이는 뷰의 유무와 공유 셰이더 여부.

### 7.4 플랫폼
- GL 컨텍스트 버전, 확장 로딩 방식, `glGetString(GL_EXTENSIONS)` 로그 유무. 확인 목록: `GL_OES_EGL_image_external_essl3`, `GL_EXT_texture_filter_anisotropic`, `GL_EXT_disjoint_timer_query`, `EGL_EXT_image_dma_buf_import(_modifiers)`, `EGL_KHR_fence_sync`.
- GL 한계: `GL_MAX_TEXTURE_SIZE`, `GL_MAX_COLOR_ATTACHMENTS`, RGBA8 밉 레벨 FBO completeness(실기 확인).
- 빌드 시스템, 셰이더 리소스 로딩 방식(파일/임베드), 설정 파일 체계.

### 7.5 분석 결과물
- Appendix C 리포트.
- "변경 영향 지도": 수정할 파일 목록과 이유, 새로 만들 파일, 건드리지 않을 파일.
- §4.6 메타데이터 전달 방식 A/B/C 중 선택과 근거.

---

## 8. 마일스톤

| 단계 | 산출물 | 종료 기준 |
|---|---|---|
| M0 분석 | Appendix C 리포트, 변경 영향 지도, 설계안(§6 준수) | 담당자 리뷰 승인 |
| M1 원인 확정·임시조치 | 단계별 ROI 덤프 도구, ISP 샤프닝 OFF 요청서, 테스트 매트(주기 5/8/12/20 mm 격자, 체커) 촬영 세트 | 모아레가 탑뷰 워프 단계에서 처음 생김을 확인(또는 반증) |
| M2 메타데이터 + 적응 SS | 캘리브레이션 툴 확장, 메타 파일 v1, `MODE_ADAPTIVE_SS`, A/B 스위치 | 중거리 LF-PSNR 개선, 프레임 시간 예산 내, 회귀 없음 |
| M3 피라미드 + 프로브 | PyramidGenerator, `MODE_PYRAMID_PROBES`, 디버그 레벨 뷰 | 근거리 포함 모아레 소거, 등방 trilinear 대비 선명도 유지, GPU 시간 ≤ 예산 |
| M4 튜닝 | N_max/bias 스윕 결과, HW 이방성 A/B(확장 있을 때), 시간축 교대 실험 | 주행 영상에서 깜빡임 지표 개선 |
| M5 검증·문서 | 합격 리포트, 운영 문서, IP 검토 결과 | 릴리스 판단 |

---

## 9. 검증 계획과 합격 기준

- **기준 영상**: 오프라인에서 같은 캘리브레이션으로 16×16 슈퍼샘플(또는 EWA) 탑뷰를 만든다(`moire_demo.py`의 `ground_truth` 방식 참고).
- **지표**: LF-PSNR(σ=2 px 저역 통과 후 PSNR)을 근거리(<1.5 m)/중거리(1.5~3.5 m)/원거리로 나눠 계산; 전대역 PSNR(과도 블러 감지); 주차선/소형 물체 대비; 주행 영상의 픽셀별 시간 분산(깜빡임); 심 불연속; 지연; 프레임 드롭; GPU 시간; CPU/열.
- **테스트 장면**: 격자 매트(주기 5·8·12·20 mm), 체커 매트, 배수 그레이팅, 아스팔트/자갈, 주차선, 소형 장애물, 심 주변, 정지·서행.
- **합격(초안)**: 근거리 LF-PSNR ≥ 현재 +5 dB, 전대역 PSNR 등방 trilinear 대비 ≥ +2 dB, 깜빡임 분산 ≥ 50 % 감소, GPU 시간 증가 ≤ 30 %, 심에서 좌우 블러 차이 없음. 실측 후 조정.
- **GPU 카운터(벤더 도구 사용 가능 시)**: 텍스처 유닛 점유율, L1 miss %(≤10 % 권고), 상위 레벨 사용 비율.

---

## 10. 참고 자료

- Khronos: `OES_EGL_image_external`, `OES_EGL_image_external_essl3`, `OES_EGL_image`, `EXT_EGL_image_storage`, `EXT_texture_filter_anisotropic`, OpenGL ES 3.2 §8.14.1 (LOD), §feedback loops.
- Arm GPU Best Practices Developer Guide §8.4–8.5 (텍스처 필터링 비용). Qualcomm Adreno Game Developer Guide (이방성 평균 비용 <2×). Imagination PowerVR mipmapping 권고.
- Heckbert, *Fundamentals of Texture Mapping and Image Warping* (1989) ch.3 — 공간 가변 리샘플링 필터, EWA, "블러를 택하라".
- McCormack et al., *Feline* (SIGGRAPH 1999); Mavridis & Papaioannou, *High Quality Elliptical Texture Filtering on GPU* (I3D 2011) — 장축 프로브 근사.
- Valeo/Connaught US 2021/0398256, US 11302062 — 서라운드뷰 안티에일리어싱 선행 특허.
- TI Mesh LDC 자료("no internal anti-aliasing filter"), TI VPAC MSC(참고: 제약 해제 시 변형).
- OpenCV `INTER_AREA` 문서("moire-free" 축소), `remap`은 INTER_AREA 미지원.
- 동반 문서: 리서치 리포트(기법 비교·실험), 해설 페이지(개념), 독서 로드맵. 스크립트: `minification_map.py`, `moire_demo.py`.

---

## Appendix A. 피라미드 생성 셰이더

```glsl
// 공통 버텍스 셰이더 (풀스크린 삼각형)
#version 300 es
out vec2 vUV;
void main() {
    vec2 p = vec2((gl_VertexID == 1) ? 3.0 : -1.0, (gl_VertexID == 2) ? 3.0 : -1.0);
    vUV = 0.5 * p + 0.5;
    gl_Position = vec4(p, 0.0, 1.0);
}

// 패스 0: 외부 텍스처 → 피라미드 레벨 0 (원본의 1/2)
#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision mediump float;
uniform samplerExternalOES uSrc;
uniform vec2 uSrcTexel;            // (1/W_src, 1/H_src)
in vec2 vUV;                       // 목적지 텍셀 중심의 정규화 좌표 (= 소스에서도 같은 정규화 위치)
out vec4 fragColor;
void main() {
    vec2 o = 0.75 * uSrcTexel;
    fragColor = 0.25 * (texture(uSrc, vUV + vec2(-o.x, -o.y)) + texture(uSrc, vUV + vec2( o.x, -o.y))
                      + texture(uSrc, vUV + vec2(-o.x,  o.y)) + texture(uSrc, vUV + vec2( o.x,  o.y)));
}

// 패스 1, 2: 피라미드 레벨 i-1 → 레벨 i  (BASE_LEVEL = MAX_LEVEL = i-1 로 묶은 뒤)
#version 300 es
precision mediump float;
uniform sampler2D uSrc;
uniform vec2 uSrcTexel;            // 레벨 i-1 의 텍셀 크기
in vec2 vUV;
out vec4 fragColor;
void main() {
    vec2 o = 0.75 * uSrcTexel;
    fragColor = 0.25 * (texture(uSrc, vUV + vec2(-o.x, -o.y)) + texture(uSrc, vUV + vec2( o.x, -o.y))
                      + texture(uSrc, vUV + vec2(-o.x,  o.y)) + texture(uSrc, vUV + vec2( o.x,  o.y)));
}
```

호스트 측 골격:

```c
// init (카메라당)
glGenTextures(1, &pyr); glBindTexture(GL_TEXTURE_2D, pyr);
glTexStorage2D(GL_TEXTURE_2D, 3, GL_RGBA8, W/2, H/2);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 2);

// build (프레임마다)
for (level = 0; level < 3; ++level) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, pyr, level);
    glViewport(0, 0, (W/2) >> level, (H/2) >> level);
    if (level == 0) { use(progExt);  bind(GL_TEXTURE_EXTERNAL_OES, camTex); setTexel(1.0/W, 1.0/H); }
    else {
        use(progPyr); bind(GL_TEXTURE_2D, pyr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, level-1);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL,  level-1);
        setTexel(1.0/((W/2) >> (level-1)), 1.0/((H/2) >> (level-1)));
    }
    glDrawArrays(GL_TRIANGLES, 0, 3);
}
glBindTexture(GL_TEXTURE_2D, pyr);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL,  2);
// FBO completeness 는 init 시 레벨별로 한 번 확인
```

## Appendix B. 메타데이터 계산 참조 구현 (Python)

```python
import numpy as np

def tri_meta(p, q, W, H, n_max=8, bias=0.3):
    """p: 삼각형 꼭짓점 출력 좌표 (3,2) [출력 px], q: 대응 원영상 좌표 (3,2) [원영상 px].
    반환: (lod, major_u, major_v, N, s_max, s_min, theta)"""
    P = np.stack([p[1]-p[0], p[2]-p[0]], axis=1)          # 2x2 열벡터
    Q = np.stack([q[1]-q[0], q[2]-q[0]], axis=1)
    J = Q @ np.linalg.inv(P)                              # 원영상 px / 출력 px
    a, b, c, d = J[0,0], J[0,1], J[1,0], J[1,1]
    A, B, C = a*a + b*b, a*c + b*d, c*c + d*d
    L = (A + C + np.sqrt((A - C)**2 + 4*B*B)) / 2
    s_max = np.sqrt(max(L, 1e-12))
    s_min = abs(a*d - b*c) / s_max
    theta = 0.5 * np.arctan2(2*B, A - C)
    if s_max <= 1.0:                                      # 확대 구간
        return (0.0, 0.0, 0.0, 1, s_max, s_min, theta)
    m = max(1.0, s_min, s_max / n_max)
    lod = np.log2(m) + bias
    N = int(np.clip(np.ceil(s_max / m - 1e-6), 1, n_max))
    major = s_max * np.array([np.cos(theta), np.sin(theta)])
    return (lod, major[0] / W, major[1] / H, N, s_max, s_min, theta)
```

정규화 UV로 저장된 메시라면 `q = uv * (W, H)`. 검증: `moire_demo.py`의 `svd2`/`method_aniso`와 같은 값이 나와야 한다(그 스크립트는 픽셀별 유한차분 자코비안 사용).

## Appendix C. 런타임 Capability 리포트 템플릿 (M0 산출물)

```
[GL/EGL]
GL_VENDOR / GL_RENDERER / GL_VERSION / GL_SHADING_LANGUAGE_VERSION:
EGL version / EGL extensions:
GL extensions (전체 목록 첨부) — 확인 항목:
  GL_OES_EGL_image_external_essl3: yes/no
  GL_EXT_texture_filter_anisotropic: yes/no (MAX_TEXTURE_MAX_ANISOTROPY_EXT = )
  GL_EXT_disjoint_timer_query: yes/no
  EGL_EXT_image_dma_buf_import / _modifiers: yes/no
  EGL_KHR_fence_sync / EGL_ANDROID_native_fence_sync: yes/no
GL_MAX_TEXTURE_SIZE / GL_MAX_COLOR_ATTACHMENTS:
RGBA8 level>0 attachment FBO completeness (실측):

[카메라 입력]
카메라 수 / 해상도 / FPS:
EGLImage 생성 API·타깃·속성(FOURCC, 플레인, 스트라이드, 오프셋, 색공간/범위/크로마 위치):
GL 타깃 (EXTERNAL_OES / TEXTURE_2D):
acquire/release 메커니즘, 버퍼 풀 크기, 프레임 시퀀스 번호:
CPU 매핑 가능 여부와 캐시 규칙:

[메시/캘리브레이션]
메시 생성 파일·함수:
버텍스 포맷 / 인덱스 여부 / 토폴로지 / 버텍스 간격(출력 px) / 카메라별 메시 여부:
월드→영상 매핑 함수(모델, 파라미터 소스):
산출물 포맷·버전·로더:
심/블렌딩 방식:

[셰이더/렌더]
탑뷰 VS/FS 파일, GLSL 버전, precision, 샘플러 타입, 블렌딩/감마 위치:
렌더 패스 구성, FBO 사용, 출력 해상도, 목표 FPS, 스레드/컨텍스트 구조:
기존 프로파일링/디버그 훅:
3D 보울 뷰 유무:

[제안]
메타데이터 전달 방식(A/B/C)과 근거:
변경 영향 지도(수정/신규/불변 파일):
예상 리스크:
```

## Appendix D. 결정 기록 (ADR 요약)

| # | 결정 | 대안 | 근거 |
|---|---|---|---|
| D1 | 프리필터로 해결, 사후 필터 배제 | FFT 노치 등 | 앨리어싱 비가역, 주파수 가변 |
| D2 | 레벨 0은 외부 텍스처, 1/2 이하만 소유 피라미드 | 풀 복사 + GenerateMipmap | 트래픽 1/4, 제로카피 유지, 필터 제어 |
| D3 | 삼각형별 정적 메타데이터 | 프래그먼트 dFdx 실시간 | 정적 매핑, 전단 과소평가 회피, ALU 절감 |
| D4 | 장축 프로브(bounded aniso), N_max 8, bias +0.3 | 등방 trilinear / HW AF | 이방성 구간 선명도, 비용 상한, 이식성 |
| D5 | 4탭 양선형 = 4×4 이항 커널로 레벨 생성 | 2×2 박스, 분리 가우시안 5탭 | 동일 비용에 −9 dB vs −3 dB |
| D6 | 프래그먼트/FBO 패스로 피라미드 생성 | 컴퓨트 | 타일 GPU 친화, 코어 이미지 포맷 제약 회피 |
| D7 | 1단계 적응 SS는 임시조치, 간격 ≤2 텍셀 규칙 | 넓은 오프셋 소수 탭 | 소수 넓은 탭은 필터가 아니라 성긴 샘플링 |
| D8 | ISP 샤프닝 OFF는 튜닝 요청으로 처리 | 앱에서 제어 | 벤더 API 금지 |
| D9 | NEON은 오프라인 계산 한정 | NEON 피라미드 | EGLImage CPU 포인터 부재 |
| D10 | IP 검토 요청(Valeo 특허) | — | 밀도 게이팅 유사성 |
