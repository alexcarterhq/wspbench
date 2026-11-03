"""Persistent numeric workspaces with an OS isolation boundary provided by Docker."""
from __future__ import annotations

import asyncio
import json
import subprocess
import uuid

IMAGE = "docker.io/library/wspbench-worker:0.1"


class WorkspaceFailure(RuntimeError):
    pass


def docker_status(image: str = IMAGE) -> dict:
    try:
        result = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True, timeout=20)
        if result.returncode:
            return {"available": False, "reason": "Start Docker Desktop or a Docker engine"}
        check = subprocess.run(["docker", "image", "inspect", image, "--format", "{{.Id}}"], capture_output=True, text=True, timeout=15)
        return {"available": True, "version": result.stdout.strip(), "image": image,
                "image_digest": check.stdout.strip() if check.returncode == 0 else None,
                "image_ready": check.returncode == 0}
    except FileNotFoundError:
        return {"available": False, "reason": "Install and start Docker, then build docker/Dockerfile"}
    except subprocess.TimeoutExpired:
        return {"available": False, "reason": "Docker engine timed out; wait for it to become responsive"}


class DockerWorkspace:
    def __init__(self, limits, image: str = IMAGE):
        self.limits = limits
        self.image = image
        self.name = "wspbench-" + uuid.uuid4().hex
        self.process = None
        self.cpu_seconds = 0.0
        self.calls = 0
        self.lock = asyncio.Lock()
        self.failed = False
        self.rpc_timeout = 35

    async def start(self, weights: dict):
        command = ["docker", "run", "--rm", "-i", "--name", self.name,
                   "--network", "none", "--read-only", "--cap-drop", "ALL",
                   "--security-opt", "no-new-privileges", "--pids-limit", "64",
                   "--memory", f"{self.limits.memory_mb}m", "--memory-swap", f"{self.limits.memory_mb}m",
                   "--cpus", "1", "--ulimit", f"cpu={self.limits.cpu_seconds}:{self.limits.cpu_seconds}",
                   "--tmpfs", "/tmp:rw,noexec,nosuid,size=16777216",
                   "--tmpfs", "/work:rw,noexec,nosuid,size=16777216,uid=65534,gid=65534",
                   self.image]
        self.process = await asyncio.create_subprocess_exec(*command, stdin=asyncio.subprocess.PIPE,
                         stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=32 * 1024 * 1024)
        await self._request({"op": "init", "weights": {k: v.tolist() for k, v in weights.items()}})
        return self

    async def _request(self, payload: dict) -> dict:
        if self.failed:
            raise WorkspaceFailure("Workspace invalidated after a failed RPC")
        if self.process is None or self.process.returncode is not None:
            raise RuntimeError("Workspace is not running")
        raw = json.dumps(payload, allow_nan=False).encode() + b"\n"
        if len(raw) > 32 * 1024 * 1024:
            raise ValueError("Workspace request exceeds 32 MiB")
        try:
            async with self.lock:
                self.process.stdin.write(raw)
                await self.process.stdin.drain()
                response = await asyncio.wait_for(self.process.stdout.readline(), timeout=self.rpc_timeout)
        except (asyncio.TimeoutError, BrokenPipeError, ConnectionError) as exc:
            self.failed = True
            await self.close()
            raise WorkspaceFailure("Workspace RPC failed; worker terminated to prevent stale responses") from exc
        if not response:
            self.failed = True
            await self.close()
            raise WorkspaceFailure("Workspace terminated, possibly due to its CPU/memory limit")
        result = json.loads(response)
        self.cpu_seconds += result.get("cpu_seconds", 0)
        if self.cpu_seconds > self.limits.cpu_seconds:
            self.failed = True
            await self.close()
            raise WorkspaceFailure("Workspace CPU limit exceeded")
        return result

    async def execute(self, code: str) -> dict:
        if self.calls >= self.limits.python_calls:
            raise ValueError("Python call allowance exhausted")
        self.calls += 1
        return await self._request({"op": "exec", "code": code})

    async def weights(self) -> dict:
        import numpy as np
        result = await self._request({"op": "weights"})
        if "error" in result:
            raise ValueError(result["error"])
        return {k: np.asarray(v, dtype=np.float32) for k, v in result["weights"].items()}

    async def close(self):
        if self.process is not None:
            if self.process.stdin:
                self.process.stdin.close()
            try:
                await asyncio.wait_for(self.process.wait(), 3)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            self.process = None
        cleanup = await asyncio.create_subprocess_exec("docker", "rm", "-f", self.name,
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await cleanup.wait()
