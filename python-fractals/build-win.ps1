echo "Creating executable"

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
Compress-Archive @compress
