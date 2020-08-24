echo "Building C-library"

cd c-fractals
C:/ProgramData/chocolatey/lib/mingw/tools/install/mingw64/bin/mingw32-make.exe


echo "Copying libraries to the python project"

Copy-Item libfractal.dll ../python-fractals/pyfractals/resources/libfractal_Windows.dll
Copy-Item C:/ProgramData/chocolatey/lib/mingw/tools/install/mingw64/bin/libwinpthread-1.dll ../python-fractals/pyfractals/resources/libwinpthread-1.dll
Copy-Item libfractal.dll libfractal_Windows.dll

echo "Creating executable"

cd ../python-fractals

try {C:/Python38/Scripts/pyinstaller fractals.spec}
catch {pyinstaller fractals.spec}

try {Remove-Item "dist/PyFractals/d3dcompiler_47.dll"} catch {}
try {Remove-Item "dist/PyFractals/opengl32sw.dll"} catch {}
try {Remove-Item "dist/PyFractals/Qt5Quick.dll"} catch {}
try {Remove-Item "dist/PyFractals/Qt5Qml.dll"} catch {}
try {Remove-Item "dist/PyFractals/libGLESv2.dll"} catch {}
try {Remove-Item "dist/PyFractals/Qt5Network.dll"} catch {}
try {Remove-Item "dist/PyFractals/Qt5QmlModels.dll"} catch {}
try {Remove-Item "dist/PyFractals/Qt5Svg.dll"} catch {}
try {Remove-Item "dist/PyFractals/Qt5WebSockets.dll"} catch {}

$compress = @{
  Path = "dist/PyFractals/"
  CompressionLevel = "Fastest"
  DestinationPath = "PyFractals.zip"
}
Compress-Archive -Force @compress

cd ..
