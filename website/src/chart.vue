<template>
	<div id="graph" style="width:1700px;height:900px;margin:0 auto;"></div>
</template>
<script>
	import echarts from 'echarts';
	import $ from 'jquery';
	import tool from './tool.js';
	
	export default {
		props: ["account", "date"],
		data: function () {
			return {
				chart: null,
				option: {
					animation: false,
					title: {
						text: ''
					},
					tooltip: {
						type: 'item',
					},
					textStyle: {
						fontSize: 20,
						fontFamily: "Microsoft YaHei",
					},
					series: [{
						type: 'graph',
						layout: 'force',
						roam: true,
						edgeSymbol: ['none', 'arrow'],
						edgeSymbolSize: 24,
						label: {
							show: true,
							position: 'top',
						},
						force: {
							repulsion: 4000,
						},
						nodes: [],
						edges: [],
						lineStyle: {
							color: 'black',
							curveness: 0.2,
							opacity: 0.8,
						},
					}],
				}
			}
		},
		methods: {
			draw: function () {
				var chart = this.chart;
				var option = this.option;
				option.title.text = this.account + '\t' + this.date;
				$.get('http://127.0.0.1:8888/cf', {account:this.account, date:this.date}, 
				function (data) {
					data = JSON.parse(data);
					var nodes = new Array();
					var edges = new Array();
					for (var i=0; i<data.length; i++) {
						var row = data[i];
						if (!tool.is_in(nodes,row[0])) {
							nodes.push(row[0]);
						}
						if (!tool.is_in(nodes,row[1])) {
							nodes.push(row[1]);
						}
						edges.push({
							source: row[0],
							target: row[1],
							value: row[2],
							lineStyle: {
								normal: {
									width: tool.line_width(row[2]),
								}
							},
							label: {
								show: true,
								position: 'middle',
								formatter : '{c}',
							},
						});
					}
					for (var i=0; i<nodes.length; i++) {
						nodes[i] = {
							name: nodes[i],
							symbolSize: 26,
						};
					}
					option.series[0].nodes = nodes;
					option.series[0].edges = edges;
					chart.setOption(option);
				});
			}
		},
		mounted: function () {
			this.chart = echarts.init(document.getElementById("graph"));
			this.draw();
		},
		watch: {
			account: 'draw',
			date: 'draw'
		}
	}
</script>