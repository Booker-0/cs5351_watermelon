const { route } = require("./routeConfig");
const controller = require("../controller/gitRepo");
const gitRepo = "gitRepo";

/** List All git repo*/
module.exports.getAllRepo = route.get(`/${gitRepo}`, controller.getAllRepo);
