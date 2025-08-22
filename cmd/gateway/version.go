// In file: cmd/gateway/version.go
package main

import (
	"fmt"
	"runtime"
)

var (
	version   = "dev"
	buildDate = "unknown"
	gitCommit = "unknown"
)

type BuildInfo struct {
	Version, BuildDate, GitCommit, GoVersion, Platform string
}

func GetBuildInfo() BuildInfo {
	return BuildInfo{
		Version:   version,
		BuildDate: buildDate,
		GitCommit: gitCommit,
		GoVersion: runtime.Version(),
		Platform:  fmt.Sprintf("%s/%s", runtime.GOOS, runtime.GOARCH),
	}
}