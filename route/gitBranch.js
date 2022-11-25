const { route } = require("./routeConfig");
const controller = require("../controller/gitBranch");
const gitbranch = "gitbranch";

/** List All git branches*/
module.exports.branchList = route.get(`/${gitbranch}/:repo`, controller.getBranchList);

module.exports.branchCommitInfo = route.get(`/${gitbranch}/:repo/:branch`, controller.getBranchInfo);
