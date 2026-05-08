$GCC = "C:\msys64\mingw64\bin\gcc.exe"
$MPIEXEC = "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"
$SRC = "d:\NUCES\SEM 6\PDC\Ass\q2.c"
$OUT = "d:\NUCES\SEM 6\PDC\Ass\q2.exe"
$INC = "C:\msys64\mingw64\include"
$LIB = "C:\msys64\mingw64\lib"

Write-Host "--- Compiling $SRC ---"
& $GCC $SRC -o $OUT -I $INC -L $LIB -lmsmpi

if ($LASTEXITCODE -eq 0) {
    Write-Host "--- Compilation Successful ---"
    Write-Host "--- Running $OUT with 8 processes ---"
    & $MPIEXEC -np 8 $OUT
} else {
    Write-Host "--- Compilation Failed ---"
}
