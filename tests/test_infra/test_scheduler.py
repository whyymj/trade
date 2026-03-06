#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试定时任务调度器
"""

import pytest
from scheduler.app import scheduler


def test_scheduler_jobs():
    """测试定时任务注册"""
    jobs = scheduler.get_jobs()
    job_ids = [job.id for job in jobs]
    assert "sync_funds" in job_ids
    assert "auto_train" in job_ids


def test_scheduler_job_names():
    """测试定时任务名称"""
    jobs = scheduler.get_jobs()
    job_names = {job.id: job.name for job in jobs}
    assert job_names["sync_funds"] == "基金数据同步"
    assert job_names["auto_train"] == "LSTM自动训练"


def test_scheduler_job_triggers():
    """测试定时任务触发器"""
    jobs = scheduler.get_jobs()
    job_triggers = {job.id: job.trigger for job in jobs}
    assert "sync_funds" in job_triggers
    assert "auto_train" in job_triggers


def test_scheduler_not_running():
    """测试调度器初始状态（未启动）"""
    assert not scheduler.running
