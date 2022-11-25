const { Octokit } = require("@octokit/core");
const { createAppAuth } = require("@octokit/auth-app");

module.exports.gitOperation = new Octokit({
    authStrategy: createAppAuth,
    auth: {
      installationId: 29556179,
      appId: 241264,
      privateKey: `-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA23GmDtDpQhqeZooTpGPP3BI3sleB8XKGgyuiqdgPE6KaXbmU
      Q0qVNsHyXIIbAou02lQUkniEN/txAGO+gkigi0hD63hyq7R2BzZvSSpt3OsuZW0n
      1flM/i4CttMOr2LCK2b0BwfM9wdnE8ZB99vGcLssl9T8Q7gRxrAdCRnlwtTb6Onz
      hOJvZo2jEFwG7tPLQeoZ7BiM3DJr1YIv7Vb1VwrGgLKWYbGN90uPqueCkLy9GwOA
      LNd/XkLuDnElykO2zSFRXRM5AJfWWmIYXScg6AJGhTHEM4wOid7m+bJ2lxRG/Llk
      G078EQNVkPOumISul6utFGLSsjFXNXO0FTc5cwIDAQABAoIBAAJuUd6cJdlbniGY
      qfsKOAVCFmfdXE0mbaMKWrTy9cfL51XedmwAaYK/x0WLE22Fyq0YfVnaB+zu1iOg
      9z1OPKkOVeJH5U1NpHkT+0ueMrVdzZfGC8jtCBNQwyrvT5xVxbzhWRay2WxrIpMv
      gQ+T2bDqAeZ0r464fUnsLYYqSUdNOxoUxvBCXd4PYYbsO4TrfGyy498WE1f0ErHJ
      2yJj5fnf6rC1guROSdXDv/V+AXcOeRQ8+6g+xNp9DpodpLIq2pSDnBm1v6t+dAD5
      +NBbxr3ZQv7IZvUhOa7nbrpYn9/jm4/ZwNPMrC7rT+YZInLciJDlTgS7elFjteXa
      YW2nrRECgYEA7T7QAJPIoGCRV0MOIIht9uQysxt4UtN9lmjG/foZlKAZwcUowN9s
      nDZXrfSj36vjPam17i3QDfNj0qA0ISBIuEJn22OhcqQWyOsLbJ4dZ8mUayiZBJuI
      jPCVSw7XxBgZ+u6Pm3OkK/pkMmV+DC7NVLm8t7ChRKxlaVlkNODBb7cCgYEA7MqV
      te0dSknz+pfKkhIe1QGSh0OZBdYinyun1BEzYBbkIAQbQMTZsvgbeOY7ioqZ+puY
      XKupQm25WF9Xh+d+nAgwbcBfqE2mm2NbrnNeqyV69XCOQu6xaZNAkh9FHBmbSLhd
      u2nIcFIKC2ZVtMAdG01n77wWhIx8fS0QdJ0mjCUCgYEAw0G5C2CV9HjF5d3IWLow
      VsyFdaecJf7uE8Z0UD9wokQKLtJHskWwK/kFvKPl44aiZfOxSi/mVjUE6Sr0/HaH
      oy70LzoWfDXUktPv+RtA9FSRlIg7N/GSNv0iwj2bE0cKyt1gz/4jFhbkNB4X2YsD
      b0HWsg/rCowggs4RjPuV+I0CgYEArR4nekyS/3812t4jAcwxsnVl1XK8a6H0yf42
      wzqYHwZdXnLiIeZJayktnKRmn5FZpfkf2ZC/PIvP2CZMblX3IMhz76mXxgqPZker
      /cznR6UtUkgqGhE8r/0yViJ6emLWPsJb9OsP2d6A7Xix7GYQYaej94fBxsKXOUU7
      JKbISekCgYAerQBHhWNjvtDMEL29m023DYhsGW/ISy34Hg82wnRKWyIAcD0TiD1b
      EgGULCYOsXaH+B/8tSFIFv6OP6ecVUzIzXP0AsOlhpXysYuOV1RK+PDwMelN8+CE
      jrz6LKF/3t+h6NBrwBOUimSUveWan13P4hCJ2SQ8yyhHZYKs/uXAdw==\n-----END RSA PRIVATE KEY-----`,
      clientId: "Iv1.440d7e7c23717b0e",
      clientSecret: "451a80c590de54727b6124272892227d0bab1fd3",
    },
  });