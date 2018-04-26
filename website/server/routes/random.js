var express = require('express');
var router = express.Router();

var mysql = require('mysql');

Date.prototype.format = function () {
	return this.getFullYear() + '-' + (this.getMonth()+1) + '-' + (this.getDate());
};

router.get('/', function(req, res) {
	var conn = mysql.createConnection({
		host: '219.224.169.45',
		user: 'lizimeng',
		password: 'codegeass',
		database: 'capital_flow'
	});
	conn.connect();
	conn.query('select account,date from basic order by rand() limit 1', function (err, result) {
		if (err) {
			console.log(err);
			return ;
		}
		var data = {
			account: result[0].account,
			date: result[0].day.format()
		};
		res.writeHead(200, {'Access-Control-Allow-Origin': '*'});
		res.end(JSON.stringify(data));
		conn.end();
	});
});

module.exports = router;
