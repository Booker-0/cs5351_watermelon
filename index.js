// const express = require('express');
// const app = express();
// const { createAppAuth, createOAuthUserAuth } = require("@octokit/auth-app");
// const { Octokit } = require("@octokit/core");
// const { gitOperation } = require('./model/gitModel');

const {route: app} = require("./route/routeConfig");
const swaggerUi = require('swagger-ui-express');
const swaggerDocument = require('./swagger');

require("./route/gitBranch");
require("./route/gitRepo");
require("./route/project");

app.use('/api/docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument));

const port = 3000;

app.listen(port, () => {
    console.log(`Example app listening on port ${port}`)
  })
