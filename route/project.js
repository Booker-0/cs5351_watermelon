const { route } = require("./routeConfig");
const controller = require("../controller/project");
module.exports.connectProjectWithGithub = route.post(`/projectConnection/:projectId/:githubName`, controller.connectProjectWithGithub);

module.exports.getGithubNameByProjectId = route.get(`/projectConnection/githubName/:projectId`, controller.getProjectGithubName);

module.exports.updateProjectbyId = route.put(`/projectConnection/update/:projectId/:githubName`, controller.updateProjectbyId);

module.exports.readTimer = route.get(`/projectConnection/timerInfo/:projectId/:taskId`, controller.getTimerInfo);

module.exports.createTimer = route.post(`/projectConnection/timerInfo/:projectId/:taskId/:time`, controller.createTimer);

module.exports.updateTimer = route.put(`/projectConnection/timerInfo/:projectId/:taskId/:time`, controller.updateTimer);

module.exports.getMachineLearningInputTraining = route.get(`/machineLearning/Info`, controller.getMachineLearningInfo);

module.exports.getMachineLearningInputTrainingByids = route.get('/machineLearning/Info/:projectId/:taskId', controller.getMachinelearningInfoById);

module.exports.getProjectIdByCode = route.get('/projectId/:projectCode', controller.getProjectIdByCode);

module.exports.connectBranchToTask = route.post('/taskConnection/:taskId/:branchName', controller.connectBranchToTask);

module.exports.getBranchNameByTaskId = route.get('/taskConnection/:taskId', controller.getBranchNameByTaskId);

module.exports.trainMachineLearningModel = route.post('/train/ML', controller.trainMachineLearningData);

module.exports.predictMachineLearningModel = route.post('/predict/ML', controller.inferenceMachineLearningData);
