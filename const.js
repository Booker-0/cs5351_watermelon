const errorMessage = {
    developer: {
        parameterNull: "Please check your input parameter"
    },
    client: {
    }
}

const statusCode = {
    100: "what happen",
}

module.exports = {
    success: (message, data) => {
        if (message === null || data === null) throw(errorMessage.developer.parameterNull);
        return {status: 200,message: message, data:data};
    },
    error: (status, message, data) => {
        if (status === null || message === null || data === null)
            throw(errorMessage.developer.parameterNull);

        return {status: status,message: message, data:data};
    },
};