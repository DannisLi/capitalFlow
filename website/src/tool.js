exports.is_in = function (arr, x) {
	for (var i=0; i<arr.length; i++) {
		if (arr[i]==x)
			return true;
	}
	return false;
}

exports.line_width = function (v) {
	const min = 1, max=10;
	var tmp = Math.sqrt(v) / 20;
	if (tmp>=min && tmp<=max)
		return tmp;
	else if (tmp<min)
		return min;
	else
		return max;
}