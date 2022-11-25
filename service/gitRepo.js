const { gitOperation } = require("../model/gitModel");

module.exports.getAllRepository = async () => {
    try {
      
      const res = await gitOperation.request("GET /installation/repositories", {});
      const { data: {repositories} } = res;
      let formattedRepositories = repositories.map(repo => ({id: repo.id, name: repo.name, node_id: repo.node_id}));
      return formattedRepositories;
    } catch (err) {
      console.log(err);
      throw new Error(err);
    }
  }

