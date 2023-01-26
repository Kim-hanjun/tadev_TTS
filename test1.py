import numpy as np
import soundfile as sf
import webrtcvad


def float_to_byte(sig):
    # float32 -> int16(PCM_16) -> byte
    return float2pcm(sig, dtype="int16").tobytes()


def float2pcm(sig, dtype="int16"):
    """Convert floating point signal with a range from -1 to 1 to PCM.
    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.
    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.
    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.
    See Also
    --------
    pcm2float, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != "f":
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in "iu":
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def vad_check(target):
    vad = webrtcvad.Vad()
    vad.set_mode(0)
    array, sampling_rate = sf.read(target)
    silence_window = 1.0
    epdMax = 10
    silence_window = int(max(min(silence_window * 1000, 2000), 300) / 10)  # 100
    epdMax = max(min(epdMax, 20), 1) * 1000  # 10000
    epd_startMargin = 50
    sample_per_frame = int(sampling_rate / 100)  # 160
    epd_predict = []
    state = 0
    epdStartX = -1
    currentFrameX = 0
    for idx in range(currentFrameX * sample_per_frame, len(array) - (sample_per_frame * 3), sample_per_frame):
        ret = vad.is_speech(float_to_byte(array[idx : idx + sample_per_frame * 3]), sampling_rate)
        epd_predict.append(ret)
        if state == 0:
            if np.sum(epd_predict[-epd_startMargin:]) > int(epd_startMargin * 0.8):
                # 소리가 없다가, 소리가 있음이 starMargin 기준 80%가 넘어가면
                state = 1
                if len(epd_predict) >= epd_startMargin:
                    # 음성 발화 판단의 길이가 임계치보다 크거나 같은경우에는
                    for i in range(epd_startMargin):
                        # 가장 최근 데이터만 발화로 사용하도록 한다. (이후에 중앙값을 이용한 임계치 판단을 위해 꼭 필요한 과정이다.)
                        epd_predict[-i] = True
                else:
                    # 너무 초반에 발화가 80%를 넘어가는 경우, 그냥 모든 음성을 다 사용하는 셈 치도록 한다.
                    for i in range(len(epd_predict)):
                        epd_predict[i] = True
                # epdStartX는 50개까지 조용하다가, 이후에 발화가 감지된 index기준으로 잡음. (epdStartX를 기준으로 자르면, 50/16000초만큼(약 3밀리초) 조용하다가 발화가 시작됨.)
                # 그래서 64라고 하면, 14번째부터 50개까지 0 이다가, 64번째에 1이 나오기 시작해서 14로 잡힘.
                epdStartX = max(0, currentFrameX - epd_startMargin)
        # 앞에 다 0이었다가, 갑작스럽게 발화가 매우 짧은 프레임에서 시작되면,
        # 1로 바뀌자마자마 median이 0으로 수렴해서 발화 종료로 판단되는듯
        if state == 1 and currentFrameX > silence_window:
            """
            발화가 시작되었고 70/16000초 이후 시점이라면,
            위의 예제를 가지고 예를 들어보면 currentFrame이 75번째 시점이라고 가정했을때, 70 - 70*(75-14)/(1000/10) = 70 - int(42.7) = 28
            대략 발화가 진행되고 있을 적당한 시점을 잡아서,
            """
            epdWindow = silence_window - int(silence_window * (currentFrameX - epdStartX) / (epdMax / 10))
            # print(
            # f"{epdWindow} :: {silence_window} - int({silence_window} * {(currentFrameX - epdStartX)} / {(epdMax / 10)}) "
            # )
            # print(
            # f"{1-(sum(epd_predict[-epdWindow:]) / len(epd_predict[-epdWindow:]))} :: {len(epd_predict[-epdWindow:]) - sum(epd_predict[-epdWindow:])} / {len(epd_predict[-epdWindow:])}"
            # )
            """
                만약 발화 진행 이후 최근시점까지의 과반수가 0이거나 (중앙값을 따져봤을때 발화가 없음), 발화 시작시점으로부터 현재까지 시점이 epdMax(10000)/10 보다 크면 발화 종료로 판단.
                TODO 2번째 옵션은 발화 시작 이후, 10초 이상 이야기 하면 무조건 자르게 될 것이다. 과연 적절한 값인지 판단 필요하다.
                """
            if np.median(epd_predict[-epdWindow:]) == 0 or (currentFrameX - epdStartX) >= int(epdMax / 10):
                state = 2
                break
        currentFrameX += 1
    result = {
        "state": state,
        "state_sec": (currentFrameX * sample_per_frame) / sampling_rate,
        "vad_total_sec": len(array) / sampling_rate,
    }
    return result


target = "/home/jun/workspace/m4a_to_wav/민경녹음wav1/1.wav"
result = vad_check(target)
print(result)
