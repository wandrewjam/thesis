if [ -z $1 ]; then
	echo "script requires an input"
	exit 1;
fi

osys=$(uname)
name=$(uname -n)

if [ $osys == "Darwin" ]; then
	server="mac"
	echo "On Mac; setting MKL threads"
	export MKL_NUM_THREADS=1;
elif [ $osys == "Linux" ]; then
	server="linux"
	
	if [[ $name =~ "peak" ]]; then
		export OMP_NUM_THREADS=1;
		echo "On CHPC server; setting OMP threads"
	elif [[ $name =~ ".math.utah.edu" ]]; then
		export OPENBLAS_NUM_THREADS=1;
		echo "On Math server; setting OPENBLAS threads"
	else
		echo "uname -n returned an unexpected value"
		exit 1;
	fi
else 
	echo "uname returned an unexpected value"
	exit 1;
fi

# python motion_integration.py $1 $server
echo $name
