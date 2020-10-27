module.exports = {
    "preset": 'ts-jest',
    "roots": [
      "<rootDir>/src/", 
      "<rootDir>/tests/"
    ],
    "testMatch": [
      "**/__tests__/**/*.+(ts|tsx|js)",
      "**/?(*.)+(spec|test).+(ts|tsx|js)"
    ],
    "transform": {
      "^.+\\.(ts|tsx)$": "ts-jest"
    },
    "moduleFileExtensions": [
      "ts",
      "js",
      "json",
      "node"
    ],
    "testEnvironment": "node",
  }