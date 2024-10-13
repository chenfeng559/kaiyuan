export default [
	{
		path: "/home",
		name: "home",
		label: "数据管理",
		icon: "HomeOutlined",
		url: "/home/index",
	},
	{
		path: "/mall",
		name: "mall",
		label: "光伏健康管理",
		icon: "ShopOutlined",
		url: "/mall/index",
	},
	{
		path: "/user",
		name: "user",
		label: "光伏功率预测",
		icon: "UserOutlined",
		url: "/user/index",
	},
	{
		path: "/other",
		label: "风电",
		icon: "SettingOutlined",
		children: [
			{
				path: "/other/pageOne",
				name: "page1",
				label: "风电健康管理",
				icon: "SettingOutlined",
			},
			{
				path: "/other/pageTwo",
				name: "page2",
				label: "风电健康预测",
				icon: "SettingOutlined",
			},
		],
	},
]
