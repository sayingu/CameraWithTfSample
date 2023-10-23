# CameraWithTfSample

## 참조 문서

[개발자용 Android](https://developer.android.com/) 참조

1. [미리보기 구현](https://developer.android.com/training/camerax/preview?hl=ko)

2. [런타임 권한 요청](https://developer.android.com/training/permissions/requesting?hl=ko)

3. [CameraX + Tensorflow Lite](https://github.com/android/camera-samples/tree/main/CameraXAdvanced)

## 추가 사항
```kotlin
override fun onCreate(savedInstanceState: Bundle?) {
    // 화면켜짐유지
    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
}
```

```xml
<application>
  <activity
    android:screenOrientation="portrait"> <!-- 세로화면유지 -->
  </activity>
</application>
```
