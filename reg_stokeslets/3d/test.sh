if [ -z $1 ]; then
	echo "script requires an input"
	exit 1;
fi

osys=$(uname)

if [ $osys == "Darwin" ]; then
	server="mac"
	export MKL_NUM_THREADS=1;
elif [ $osys == "Linux" ]; then
	server="linux"
	export OPENBLAS_NUM_THREADS=1;
else 
	echo "uname returned an unexpected value"
	exit 1;
fi

python motion_integration.py $1 $server
