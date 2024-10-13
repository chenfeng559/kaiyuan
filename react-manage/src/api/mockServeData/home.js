// mock数据模拟
import Mock from "mockjs"

// 图表数据
let List = []
export default {
	getStatisticalData: () => {
		//Mock.Random.float 产生随机数100到8000之间 保留小数 最小0位 最大0位
		for (let i = 0; i < 7; i++) {
			List.push(
				Mock.mock({
					苹果: Mock.Random.float(100, 8000, 0, 0),
					vivo: Mock.Random.float(100, 8000, 0, 0),
					oppo: Mock.Random.float(100, 8000, 0, 0),
					魅族: Mock.Random.float(100, 8000, 0, 0),
					三星: Mock.Random.float(100, 8000, 0, 0),
					小米: Mock.Random.float(100, 8000, 0, 0),
				})
			)
		}
		return {
			code: 20000,
			data: {
				// 天饼图
				videoDataDay: [
					{
						name: "健康",
						value: 6000,
					},
					{
						name: "异常",
						value: 3000,
					},
					{
						name: "警告",
						value: 1500,
					},
				],

				//周饼图
				videoDataWeek: [
					{
						name: "健康",
						value: 4000,
					},
					{
						name: "异常",
						value: 8000,
					},
					{
						name: "警告",
						value: 2500,
					},
				],

				//月饼图
				videoDataMonth: [
					{
						name: "健康",
						value: 5000,
					},
					{
						name: "异常",
						value: 1000,
					},
					{
						name: "警告",
						value: 3500,
					},
				],

				// 柱状图
				userData: [
					{
						date: "周一",
						new: 5,
						active: 200,
					},
					{
						date: "周二",
						new: 10,
						active: 500,
					},
					{
						date: "周三",
						new: 12,
						active: 550,
					},
					{
						date: "周四",
						new: 60,
						active: 800,
					},
					{
						date: "周五",
						new: 65,
						active: 550,
					},
					{
						date: "周六",
						new: 53,
						active: 770,
					},
					{
						date: "周日",
						new: 33,
						active: 170,
					},
				],
				// 折线图
				orderData: {
					date: ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00"],
					data: List,
				},
				// 表格数据
				tableData: [
					{
						name: "逆变器",
						todayBuy: "正常",
						monthBuy: 98,
						totalBuy: 22000,
					},
					{
						name: "电线",
						todayBuy: "正常",
						monthBuy: 98,
						totalBuy: 24000,
					},
					{
						name: "AAAAA",
						todayBuy: "异常",
						monthBuy: 78,
						totalBuy: 65000,
					},
					{
						name: "BBBBB",
						todayBuy: "异常",
						monthBuy: 0,
						totalBuy: 0,
					},
					{
						name: "CCCCC",
						todayBuy: "异常",
						monthBuy: 0,
						totalBuy: 0,
					},
					{
						name: "DDDDD",
						todayBuy: "异常",
						monthBuy: 0,
						totalBuy: 0,
					},
				],
			},
		}
	},
}
