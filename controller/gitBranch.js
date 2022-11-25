const branchService = require("../service/gitBranch");
const { success, error } = require("../const");

module.exports.getBranchList = async (req, res) => {
    const { repo } = req.params;
    try {
    const branchCommitList = await branchService.getAllBranches(repo);
    res.status(200).send(success("get request success", branchCommitList));
    } catch(err) {
        console.log(err);
        res.status(404).send(error(404, err));
    }
}

module.exports.getBranchInfo = async (req, res) => {
    const { repo, branch } = req.params;
    try {
        const branchInfo = await branchService.getBranchAllcommitInfo(repo, branch);
        res.status(200).send(success("get branch information", branchInfo));
    } catch (err) {
        console.log(err);
        res.status(404).send(error(404, err)); 
    }
}