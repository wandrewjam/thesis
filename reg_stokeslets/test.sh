osys=$(uname)

if [[ $osys == "Darwin" ]]; then
	server="Mac"
	export MKL_NUM_THREADS=1;
elif [[ $osys == "Linux" ]]; then
	server="Linux"
	export OPENBLAS_NUM_THREADS=1;
else 
	echo "uname returned an unexpected value"
	exit 1;
fi

python tester.py $server