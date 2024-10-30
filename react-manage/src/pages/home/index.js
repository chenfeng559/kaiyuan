import React, { useEffect, useState } from "react"
import { Col, Row, Card, Table } from "antd"
import { getData } from "../../api"
import "./home.css"
import * as Icon from "@ant-design/icons"
import MyEcharts from "../../components/Echarts"

const columns = [
	{
		title: "名称",
		dataIndex: "name",
	},
	{
		title: "运行状态",
		dataIndex: "todayBuy",
	},
	{
		title: "得分",
		dataIndex: "monthBuy",
	},
	{
		title: "总运行时长",
		dataIndex: "totalBuy",
	},
]

const countData = [
	{
		name: "今日风电发电量",
		value: 1234,
		icon: "CheckCircleOutlined",
		color: "#2ec7c9",
	},
	{
		name: "今日光伏发电量",
		value: 3421,
		icon: "ClockCircleOutlined",
		color: "#ffb980",
	},
	{
		name: "今日损耗电量",
		value: 1234,
		icon: "CloseCircleOutlined",
		color: "#5ab1ef",
	},
	{
		name: "本月风电发电量",
		value: 1234,
		icon: "CheckCircleOutlined",
		color: "#2ec7c9",
	},
	{
		name: "本月光伏发电量",
		value: 3421,
		icon: "ClockCircleOutlined",
		color: "#ffb980",
	},
	{
		name: "本月损耗电量",
		value: 1234,
		icon: "CloseCircleOutlined",
		color: "#5ab1ef",
	},
]

const iconToElement = (name) => React.createElement(Icon[name])

const Home = () => {
	const userImg = require("../../assets/images/user.png")
	const [tableData, setTableData] = useState([])
	const [echartData, setEchartData] = useState({})

	useEffect(() => {
		getData().then(({ data }) => {
			const {
				tableData,
				orderData,
				userData,
				videoDataDay,
				videoDataWeek,
				videoDataMonth,
			} = data.data
			setTableData(tableData)
			const order = orderData
			const xData = order.date
			const keyArray = Object.keys(order.data[0])
			const series = []

			keyArray.forEach((key) => {
				series.push({
					name: key,
					data: order.data.map((item) => item[key]),
					type: "line",
				})
			})

			setEchartData({
				...echartData,
				order: {
					xData,
					series,
				},
				user: {
					xData: userData.map((item) => item.date),
					series: [
						{
							name: "新增用户",
							data: userData.map((item) => item.new),
							type: "bar",
						},
						{
							name: "活跃用户",
							data: userData.map((item) => item.active),
							type: "bar",
						},
					],
				},
				videoDay: {
					series: [
						{
							data: videoDataDay,
							type: "pie",
						},
					],
				},
				videoWeek: {
					series: [
						{
							data: videoDataWeek,
							type: "pie",
						},
					],
				},
				videoMonth: {
					series: [
						{
							data: videoDataMonth,
							type: "pie",
						},
					],
				},
			})
		})
	}, [])

	return (
		<Row
			gutter={[16, 16]}
			className='home'>
			<Col
				xs={24}
				md={8}>
				<Card hoverable>
					<div className='user'>
						<img
							src={userImg}
							alt='User'
						/>
						<div className='userinfo'>
							<p className='name'>Admin</p>
							<p className='access'>超级管理员</p>
						</div>
					</div>
					<div className='login-info'>
						<p>
							上次登录时间：<span>2024-9-26</span>
						</p>
						<p>
							上次登录地点：<span>广东省广州市天河区华南师范大学</span>
						</p>
					</div>
				</Card>
				<Card
					style={{ marginTop: "20px" }}
					hoverable>
					<Table
						rowKey={"name"}
						columns={columns}
						dataSource={tableData}
						pagination={false}
					/>
				</Card>
			</Col>
			<Col
				xs={24}
				md={16}>
				<Row gutter={[16, 16]}>
					{countData.map((item, index) => (
						<Col
							xs={12}
							md={8}
							key={index}>
							<Card>
								<div
									className='icon-box'
									style={{ background: item.color }}>
									{iconToElement(item.icon)}
								</div>
								<div className='detail'>
									<p className='num'>{item.value}兆瓦时</p>
									<p className='txt'>{item.name}</p>
								</div>
							</Card>
						</Col>
					))}
				</Row>
				<Row gutter={[16, 16]}>
					{echartData.order && (
						<Col span={24}>
							<div className='chart-container'>
								<MyEcharts
									chartData={echartData.order}
									style={{ height: "280px" }}
								/>
								<p className='chart-title'>分数变化情况</p>
							</div>
						</Col>
					)}
					{echartData.videoDay && (
						<Col
							xs={24}
							sm={12}>
							<div className='chart-container'>
								<MyEcharts
									chartData={echartData.videoDay}
									isAxisChart={false}
									style={{ height: "260px" }}
								/>
								<p className='chart-title'>一天内健康情况</p>
							</div>
						</Col>
					)}
					{echartData.videoWeek && (
						<Col
							xs={24}
							sm={12}>
							<div className='chart-container'>
								<MyEcharts
									chartData={echartData.videoWeek}
									isAxisChart={false}
									style={{ height: "260px" }}
								/>
								<p className='chart-title'>一周内健康情况</p>
							</div>
						</Col>
					)}
					{echartData.videoMonth && (
						<Col
							xs={24}
							sm={12}>
							<div className='chart-container'>
								<MyEcharts
									chartData={echartData.videoMonth}
									isAxisChart={false}
									style={{ height: "260px" }}
								/>
								<p className='chart-title'>一月内健康情况</p>
							</div>
						</Col>
					)}
					{echartData.user && (
						<Col
							xs={24}
							sm={12}>
							<div className='chart-container'>
								<MyEcharts
									chartData={echartData.user}
									style={{ height: "240px" }}
								/>
								<p className='chart-title'>用户活跃度</p>
							</div>
						</Col>
					)}
				</Row>
			</Col>
		</Row>
	)
}

export default Home
