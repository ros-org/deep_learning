
adb push ../build/bin/* /userdata/
adb shell "export LD_LIBRARY_PATH=/userdata/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} && /userdata/main"

