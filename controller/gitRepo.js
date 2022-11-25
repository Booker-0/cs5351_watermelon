const repooService = require("../service/gitRepo");
const { success, error } = require("../const");

module.exports.getAllRepo = async (req, res) => {
    try {
        const repositories = await repooService.getAllRepository();
        res.status(200).send(success("get repositories successfully", repositories));
    } catch(err) {
        res.status(404).send(error(404, err));
    }
}