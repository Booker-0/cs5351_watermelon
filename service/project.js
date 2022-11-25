const { sqlDb } = require("../config");
const fetch = require('node-fetch');

const tableName = 'projectGithubConnection_tbl'
const timerName = 'team_task_timer'
const getInsertProjectMapGithubSQL = (projectId, githubName, tableName, action) => {
  let sql =  `INSERT INTO ${tableName}(project_id, github_name) Values(${projectId}, '${githubName}')`;
  return sql;
}

const getReadProjectMapGithubSQL = (tableName, projectId) => {
  let sql =  `SELECT github_name FROM ${tableName} WHERE project_id = ${projectId}`;
  return sql;
}

const getUpdateProjectMapGithubSQL = (tableName, projectId, githubName) => {
  let sql = `UPDATE ${tableName} SET github_name = '${githubName}' WHERE project_id = ${projectId}`
  return sql;
}

const getUpdateTimerSQL = (tableName, projectId, taskId, stopTime) => {
  let sql = `UPDATE ${tableName} SET status = 1, stop_time = '${stopTime}', updated_date = '${new Date().toISOString().slice(0, 19).split('T')[0]}' WHERE project_id = ${projectId} AND task_id = ${taskId} AND status = 0`
  return sql;
}

const getInsertTimerSQL = (tableName, projectId, taskId, time) => {
  let sql =  `INSERT INTO ${tableName}(project_id, task_id, status, start_time) Values(${projectId}, ${taskId}, 0, '${time}')`;
  return sql;
}

const getReadTimerSQL = (tableName, projectId, taskId) => {
  let sql = `SELECT start_time, stop_time, status,  TIMESTAMPDIFF(SECOND, start_time, stop_time) as duration FROM ${tableName} WHERE project_id = ${projectId} AND task_id = ${taskId} ORDER BY timer_id DESC`;
  return sql
}

/** */
const getTotalWorkingDaySQL = (tableName, condition='') => {
  let sql = `SELECT DISTINCT task_id, project_id, updated_date FROM ${tableName} ${condition}`;
  return sql;
}

const getTotalWorkingHourSQL = (tableName, condition='') => {
  let sql = `SELECT DISTINCT task_id, project_id, updated_date FROM ${tableName} ${condition}`;
  return sql;
}

const getTotalWorkingDayAndHour = (condition='') => {
  let sql = `SELECT updated_date,task_id, project_id, SUM(TIMESTAMPDIFF(SECOND, start_time, stop_time)) as totalHour FROM team_task_timer GROUP BY updated_date, task_id, project_id
   ${condition}`
  return sql;
}

const getUpdateTaskSQL = (branchName , taskId) => {
  const sql = `UPDATE team_task SET branch_name = '${branchName}' WHERE id = ${taskId}`;
  return sql;
}

const getProjectIDByCodeSQL = (code='') => {
  const sql = `SELECT id FROM team_project WHERE code = '${code}'`

  return sql;
}

const getBranchNameSQL = (taskId) => {
  const sql = `SELECT branch_name FROM team_task WHERE id = ${taskId}`;
  return sql;
}

const getAllMachineLearningRequiredInfo = (condition='') => {
  let sql = `SELECT
    updated_date,
    team_project.name AS project_name,
    team_task.name AS task_name,
    team_member_account.name AS memeber_name,
    SUM(
        TIMESTAMPDIFF(HOUR, start_time, stop_time)
    ) AS total_hours,
    SUM(
        TIMESTAMPDIFF(MINUTE, start_time, stop_time)
    ) % 60 AS remainder_minutes
FROM
    team_task_timer
INNER JOIN
    team_project
ON
    team_project.id = team_task_timer.project_id
INNER JOIN
    team_task
ON
    team_task.id = team_task_timer.task_id
INNER JOIN
    team_member_account
ON
    team_member_account.member_code = team_task.assign_to
${condition}
GROUP BY
    updated_date,
    task_id,
    project_id
    `;

  return sql;
}

const convertStatus = (num) => {
  if (num === 1) return 'stop';

  if (num === 0) return 'running';
}

const asyncGetQuery = async (sql, sqlDb) => {
  return new Promise((resolve, reject) => {
    sqlDb.query(sql, (err, rows) => {
        if (err) reject(err);

        resolve(rows);
    });
  })
}

const formatMachineLearningData = (dataArray, projectName, employee) => {
  let strQuery = "";
  dataArray.forEach((item, i) => {
    strQuery += `data=${item}&`
  });
  strQuery += `employee_name=${employee}&`
  strQuery += `project_name=${projectName}`;

  return strQuery;
}

module.exports.getProjectIdByCode = async (projectCode) => {
  try {
    const result = await asyncGetQuery(getProjectIDByCodeSQL(projectCode), sqlDb);
    return result;
  } catch (err) {
    console.log(err);
    throw err;
  }
}

module.exports.mapeGithubToProject = async (projectId, githubName) => {
    try {
      await asyncGetQuery(getInsertProjectMapGithubSQL(projectId, githubName, tableName), sqlDb);

      return true;
    } catch (err) {
      console.log(err);
      throw err;
    }
  }

module.exports.updateGithubNameById = async (projectId, githubName) => {
  try {

    await asyncGetQuery(getUpdateProjectMapGithubSQL(tableName, projectId, githubName), sqlDb);
    return true;
  } catch (err) {
    console.log(err);
    throw err;
  }
}

module.exports.getProjectGithubNameById = async (projectId) => {
  try {
    console.log("getting the project github name by id")
    const result = await asyncGetQuery(getReadProjectMapGithubSQL(tableName, projectId), sqlDb);
    return result;
  } catch (err) {
    console.log(err);
    throw err;
  }
}

/** start time */
module.exports.createTimer = async (projectId, taskId, time) => {
  try {
    await sqlDb.query(getInsertTimerSQL(timerName, projectId, taskId, time));
    return true;
  } catch (err) {
    console.log(err);
    throw err;
  }
}

/** stop time */
module.exports.updateTimer = async (projectId, taskId, stopTime) => {
  try {
    await sqlDb.query(getUpdateTimerSQL(timerName, projectId, taskId, stopTime));
    return true;
  } catch (err) {
    console.log(err);
    throw err;
  }
}

module.exports.readTimer = async (projectId, taskId) => {
  try {
    const result = await asyncGetQuery(getReadTimerSQL(timerName, projectId, taskId), sqlDb);
    if (!result) return [{time: null, status: 'none', duration: 0}];

    const latest = result[0].status === 0 ? result.shift() : result[0];
    const duration = result?.reduce(
      (previousValue, { duration }) => previousValue + duration,
      0
    );
    return [{time: latest.status == 0 ? latest.start_time : null, status: convertStatus(latest.status), total_seconds: duration}];
  } catch (err) {
    console.log(err);
    throw err;
  }
}

module.exports.getDetailWorkingDay = async () => {
  try {
    const result = await asyncGetQuery(getAllMachineLearningRequiredInfo(), sqlDb);
    if (!result) return [{update_time: null, total_hours: 0, remainder_minutes: 0, task_id: null, project_id: null}];
    return result;
  } catch (err) {
    console.log(err);
    throw err;
  }
}

/** get detail working day by id */
module.exports.getSingleDetailWorkingDayAndTime = async (projectId, taskId) => {
  try {
    const result = await asyncGetQuery(getAllMachineLearningRequiredInfo(`WHERE project_id = ${projectId} AND task_id = ${taskId}`), sqlDb);
    if (!result) return [{update_time: null, total_hours: 0, remainder_minutes: 0, task_id: null, project_id: null}];
    return result;
  } catch (err) {
    console.log(err);
    throw err;
  }
}

module.exports.connectBranchToTask = async (branchName, taskId) => {
  try {

    console.log(branchName);
    console.log(taskId);
    const result = await asyncGetQuery(getUpdateTaskSQL(branchName, taskId), sqlDb);
    return result;
  } catch (err) {
    console.log(err);
    throw err;
  }
}

module.exports.getBranchNameByTaskId = async (taskId) => {
  try {
    const result = await asyncGetQuery(getBranchNameSQL(taskId), sqlDb);
    return result;
  } catch (err) {
    console.log(err);
    throw err;
  }
}

module.exports.trainMachineLearningData = async (pData) => {
  try {
    let { data, projectName, employee } = pData;
    console.log(data);
    data = JSON.parse(data);
    const strQuery = formatMachineLearningData(data, projectName, employee);
    const res = await fetch('http://localhost:5001/train?' + strQuery);
    const result = await res.text();
    return result;
  } catch (err) {
    console.log(err);
    throw err;
  }
}

module.exports.inferenceMachineLearningData = async (pData) => {
  try {
    // day in hour
    let { data, projectName, employee, deadline, projectEstHours} = pData;
    data = JSON.parse(data);
    let strQuery = formatMachineLearningData(data, projectName, employee);
    strQuery += `&project_est_hours=${projectEstHours}`;
    strQuery += `&deadline=${deadline}`;
    const res = await fetch('http://localhost:5001/predict?' + strQuery);
    const result = await res.text();
    return result ;
  } catch (err) {
    console.log(err);
    throw err;
  }
}
