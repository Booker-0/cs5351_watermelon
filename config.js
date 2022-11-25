// const sqlConfig = {
//     HOST: "localhost",
//     USER: "root",
//     PASSWORD: "watermelon",
//     DB: "mysql",
//     dialect: "mysql",
//     pool: {
//         max: 5,
//         min: 0,
//         acquire: 30000,
//         idle: 10000
//     }
// };

const mysql = require('mysql');
const con = mysql.createConnection({
  host: "localhost",
  user: "root",
  password: "watermelon",
  database: "teamwork",
});

module.exports.sqlDb = con;
