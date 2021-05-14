package main

import (
	"fmt"
	"os"
	"os/exec"
)

func main() {
	os.Mkdir("build", 0775)
	run("git", "submodule", "update", "--init", "--recursive")
	run("cmake", "-DONNX_ML=OFF", "..")
	run("make", "-j8")
}

func run(args ...string) {
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Dir = "build"
	cmd.Stderr = os.Stderr
	cmd.Stdout = os.Stdout
	err := cmd.Run()
	if err != nil {
		fmt.Printf("Error: %s.\n", err)
		os.Exit(1)
	}
}
