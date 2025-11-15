[app]

title = Super Resolution App
package.name = superresolution
package.domain = org.test
source.dir = .
ource.include_exts = py,png,jpg,kv,atlas,ttf
source.include_patterns = assets/*,images/*,Core/*,GUI/*,fonts/*
version = 0.1

requirements = python3,kivy==2.3.1,plyer==2.1.0,docutils==0.22.2,Pygments==2.19.2,filetype==1.2.0,certifi==2025.10.5,requests==2.32.5,urllib3==2.5.0,idna==3.11,charset-normalizer==3.4.3

orientation = portrait
fullscreen = 0
android.presplash_color = #FFFFFF
android.permissions = INTERNET, READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE
android.api = 33
android.minapi = 21
android.ndk = 25b

android.accept_sdk_license = True

android.release_artifact = apk
android.debug_artifact = apk

[buildozer]
log_level = 2
warn_on_root = 0