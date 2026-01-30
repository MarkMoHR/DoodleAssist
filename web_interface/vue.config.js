// vue.config.js
const fs = require("fs");

process.env.VUE_APP_API_URL = process.env.NODE_ENV === 'production'
	? '/demos/vtracer/backend'
	: '';

module.exports = {
	publicPath: process.env.NODE_ENV === 'production'
		? '/demos/vtracer/'
		: '/',
	// options...
	// devServer: {
	// 	// https: {
	// 	// 	key: fs.readFileSync("/etc/letsencrypt/live/tracer.cs.yale.edu/privkey.pem"),
	// 	// 	cert: fs.readFileSync("/etc/letsencrypt/live/tracer.cs.yale.edu/fullchain.pem")
	// 	// },
	// 	//host: "tracer.cs.yale.edu",
	// 	host: "localhost",
	// 	port: 8000, //443,
	// 	disableHostCheck: true
	// }

	devServer: {
		proxy: 'http://10.35.2.143:5000',
		// host: "localhost",
		// port: 8080, //443,
		disableHostCheck: true
	}
}
