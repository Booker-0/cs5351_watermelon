const { gitOperation } = require("../model/gitModel");

/** get git commit message by git branch*/
module.exports.getAllBranches = async (repo) => {
    try {
      
      let res = await gitOperation.request('GET /repos/{owner}/{repo}/branches', {
        owner: 'pwliuab',
        repo: repo
      })
      const { data: branches } = res;
      let branchCommitList = branches.map(branch => ({branch_name: branch.name, commit: branch.commit}));
      return branchCommitList;
    } catch (err) {
        throw(err);
    }
  }
  /** branch name */
  module.exports.getBranchAllcommitInfo = async (repo, branch) => {
    const branchInfo = await gitOperation.request('GET /repos/{owner}/{repo}/commits/{sha}', {
        owner: 'pwliuab',
        repo: repo,
        sha: branch
      });
      
    const { data } = branchInfo;
    
    return {files: data.files, message: data.commit.message, branchName: branch}
  }

  module.exports.getBranchesAllCommitInfo = async (repo) => {

    try {
      const branchCommitList = await getAllBranches(repo);
      const commits  = await Promise.all(branchCommitList.map(async (branch) => {
        const data = await gitOperation.request('GET /repos/{owner}/{repo}/commits/{sha}', {
          owner: 'pwliuab',
          repo: repo,
          sha: branch.branch_name
        });
        return {...data, branch_name: branch.branch_name};
      }));
  
      const modifiedInfo = commits.map(commit => {
        const { data } = commit; 
        return {files: data.files, message: data.commit.message, branchName: commit.branch_name};
      })
      return modifiedInfo;
    } catch (err) {
      console.log(err); 
      throw new Error(err);
    }
  }
