var express = require('express');
var router = express.Router();

var mysql = require('mysql');


router.get('/', function (req, res) {
	var account = req.query.account;
	var date = req.query.date;
	var conn = mysql.createConnection({
		host: '219.224.169.45',
		user: 'lizimeng',
		password: 'codegeass',
		database: 'capital_flow'
	});
	conn.connect();
	conn.query('select cf from basic where account=? and date=? limit 1', 
	[account,date], function (err, result) {
		if (err) {
			console.log(err);
			return ;
		}
		res.writeHead(200, {'Access-Control-Allow-Origin': '*'});
		res.end(result[0].cf);
		conn.end();
	});
});

module.exports = router;