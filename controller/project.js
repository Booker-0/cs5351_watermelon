const projectService = require("../service/project");
const { success, error } = require("../const");

module.exports.connectProjectWithGithub = async (req, res) => {

  try {
    const { githubName, projectId } = req.params;
    await projectService.mapeGithubToProject(projectId, githubName);
    res.status(200).send(success("create successfully", { githubName, projectId }));
  } catch (err) {
    console.log(err);
    res.status(404).send(error(404, err));
  }

}


module.exports.getProjectIdByCode = async (req, res) => {
  try {
    const { projectCode } = req.params;
    const result = await projectService.getProjectIdByCode(projectCode);
    res.status(200).send(success("get project id successfully", {projectId: result}))
  } catch (err) {
    console.log(err);
    throw err;
  }
}

module.exports.updateProjectbyId = async (req, res) => {
  try {
    const { projectId, githubName} = req.params;
    await projectService.updateGithubNameById(projectId, githubName);
    res.status(200).send(success("create successfully", { githubName,  projectId }));
  } catch (err) {
    console.log(err);
    res.status(404).send(error(404, err));
  }
}

module.exports.getProjectGithubName = async (req, res) => {
  try {
    const { projectId } = req.params;
    const githubName = await projectService.getProjectGithubNameById(projectId);
    res.status(200).send(success("get successfully",  githubName));
  } catch (err) {
    console.log(err);
    res.status(404).send(error(404, err));
  }
}


module.exports.updateTimer = async (req, res) => {
  try {
      const { projectId, taskId, time} = req.params;
      const timerInfo = await projectService.updateTimer(projectId, taskId, time);
      res.status(200).send(success("update successfully", { projectId, taskId, time }));
  } catch (err) {
    console.log(err);
    res.status(404).send(error(404, err));
  }
}

module.exports.createTimer = async (req, res) => {
  try {
    const { projectId, taskId, time} = req.params;
    const timerInfo = await projectService.createTimer(projectId, taskId, time);
    res.status(200).send(success("created successfully", { timerInfo, projectId, time}));
  } catch (err) {
    console.log(err);
    res.status(404).send(error(404, err));
  }
}


module.exports.getTimerInfo = async (req, res) => {
  try {
      const { projectId, taskId } = req.params;
      const timerInfo = await projectService.readTimer(projectId, taskId);
      res.status(200).send(success("get successfully", timerInfo));
  } catch (err) {
    console.log(err);
    res.status(404).send(error(404, err));
  }
}

module.exports.getMachineLearningInfo = async (req, res) => {
  try {
    const mlInfo = await projectService.getDetailWorkingDay();
    res.status(200).send(success("get successfully", mlInfo));
  } catch (err) {
    console.log(err);
    res.status(404).send(error(404, err));
  }
}

module.exports.getMachinelearningInfoById = async (req, res) => {
  try {
    const { projectId, taskId } = req.params;
    const mlInfo = await projectService.getSingleDetailWorkingDayAndTime(projectId, taskId);
    res.status(200).send(success("get successfully", mlInfo));
  } catch (err) {
    console.log(err);
    res.status(404).send(error(404, err));
  }
}

module.exports.getBranchNameByTaskId = async (req, res) => {
  try {
    const { taskId } = req.params;
    const branchName = await projectService.getBranchNameByTaskId(taskId);
    res.status(200).send(success("get successfully", branchName));
  } catch (err) {
    console.log(err);
    res.status(404).send(error(404, err));
  }
}

module.exports.connectBranchToTask = async (req, res) => {
  try {
    const { taskId, branchName } = req.params;
    await projectService.connectBranchToTask(branchName, taskId);
    res.status(200).send(success("connect successfully", {taskId, branchName}));
  } catch (err) {
    console.log(err);
    res.status(404).send(error(404, err));
  }
}


module.exports.inferenceMachineLearningData = async (req, res) => {
  try {
    console.log(req.body);
    const result = await projectService.inferenceMachineLearningData(req.body);
    res.status(200).send(success("predict success", result));
  } catch (err) {
    console.log(err);
    res.status(404).send(error(404, err));
  }
}

module.exports.trainMachineLearningData = async (req, res) => {
  try {
    console.log(req.body);
    const result = await projectService.trainMachineLearningData(req.body);
    res.status(200).send(success("train success", result));
  } catch (err) {
    console.log(err);
    res.status(404).send(error(404, err));
  }
}
