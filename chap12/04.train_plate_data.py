import numpy as np, cv2

def SVM_create(type, max_iter, epsilon):
    svm = cv2.ml.SVM_create()                           # SVM 객체 선언
    # SVM 파라미터 지정
    svm.setType(cv2.ml.SVM_C_SVC)             # C-Support Vector Classification
    svm.setKernel(cv2.ml.SVM_LINEAR)                    # 선형 SVM
    svm.setGamma(1)                                     # 커널 함수의 감마 값
    svm.setC(1)                                         # 최적화를 위한 C 파라미터
    svm.setTermCriteria((type, max_iter, epsilon))      # 학습 반복 조건 지정
    return svm

nsample = 140
paths = ["images/plate/%03d.png" % i for i in range(nsample)]
trainData = [cv2.imread(p, 0) for p in paths]

missing = [p for p, img in zip(paths, trainData) if img is None]
if missing:
    raise FileNotFoundError(
        "학습 영상 %d개를 읽지 못했습니다. (예: %s)\n"
        "이 예제는 images/plate/000.png ~ %03d.png 가 모두 있어야 합니다.\n"
        "학습 결과물인 SVMtrain.xml 은 이미 저장되어 있으므로, "
        "07.classify_plates.py 등은 이 스크립트를 실행하지 않아도 동작합니다."
        % (len(missing), missing[0], nsample - 1))

trainData = np.reshape( trainData, (nsample, -1)).astype("float32")
labels = np.zeros((nsample,1), np.int32)
labels[:70] = 1


print("SVM 객체 생성")
svm = SVM_create(cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6)   # SVM 객체 생성
svm.train(trainData, cv2.ml.ROW_SAMPLE, labels)             # 학습 수행
svm.save("SVMtrain.xml")                             # 학습된 데이터 저장
print("SVM 객체 저장 완료")

