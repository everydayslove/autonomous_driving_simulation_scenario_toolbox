if not exist build ( md build )
cd build 
cmake .. -DCMAKE_BUILD_TYPE=Debug -G "Visual Studio 15 2017 Win64"
pause
