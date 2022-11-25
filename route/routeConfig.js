const express = require('express');
const cors = require("cors");
const bodyParser = require("body-parser");
const app = express();
app.use((req, res, next) => {
    res.append('Access-Control-Allow-Origin', ['*']);
    res.append('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE');
    res.append('Access-Control-Allow-Headers', 'Conent-Type');
    res.append('Access-Control-Allow-Headers', '*');
    res.append('Access-Control-Expose-Headers', 'Content-Range');
    res.append('Content-Range','posts 0-20/20')
    next();
});
// parse requests of content-type - application/json
app.use (bodyParser.json());
// parse requests of content-type - application/x-www-form-urlencoded
app.use (bodyParser.urlencoded({extended:true}));




module.exports.route = app;
