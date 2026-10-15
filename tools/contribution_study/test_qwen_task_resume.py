"""Fault injection for queue recovery; no GPU/model needed."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import qwen_task_resume as recovery
from qwen_task_protocol import read, write


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, root, permanent):
        jobs = [dict(id=i, model='edge', method='fp8', benchmark='rbench', case_id=i) for i in ['bad', 'good']]
        scenes = {j['id']: {'queries': [{'id': 'q'}]} for j in jobs}
        write(root/'scenes.json', scenes)
        calls = {'bad': 0, 'good': 0}

        async def answer(root, client, job, query, media, mh):
            calls[job['id']] += 1
            if job['id']=='bad' and (permanent or calls['bad']==1):
                raise RuntimeError('Judge response failed after three attempts: bad q')
            return {'query_id': 'q', 'parsed': {'score': 3}}

        client = SimpleNamespace(close=AsyncMock(), client=SimpleNamespace(
            get=AsyncMock(return_value=SimpleNamespace(raise_for_status=lambda: None))))
        with ExitStack() as stack:
            for name, replacement in {
                'verify_manifest': lambda _: ({}, 'frozen'), 'load_jobs': lambda _: jobs,
                'await_service': AsyncMock(), 'qualify': AsyncMock(),
                'get_clients': AsyncMock(return_value=[client]), 'answer': answer,
                'make_media': lambda *args: {'path': 'fixed'},
                'summarize_video': lambda *args: {'score': 3},
            }.items():
                stack.enter_context(patch.object(recovery, name, replacement))
            stack.enter_context(patch('qwen_task_report.report', return_value={'complete': not permanent}))
            args=argparse.Namespace(root=root, passes=3, video_workers=2)
            if permanent:
                with self.assertRaisesRegex(RuntimeError, '1 invalid judge answers remain'):
                    await recovery.run(args)
            else:
                await recovery.run(args)
        return calls

    async def test_bad_answer_is_isolated_and_retried_after_good_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            calls=await self.exercise(root, False)
            self.assertEqual(calls, {'bad': 2, 'good': 1})
            self.assertEqual(read(root/'progress.json')['completed'], 2)
            self.assertTrue(read(root/'campaign.json')['complete'])
            self.assertFalse((root/'failures/bad.json').exists())
            self.assertEqual(len(list((root/'resolved_failures').glob('*.json'))), 1)

    async def test_permanent_bad_answer_is_not_zero_and_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            calls=await self.exercise(root, True)
            self.assertEqual(calls, {'bad': 3, 'good': 1})
            self.assertEqual(read(root/'progress.json')['completed'], 1)
            self.assertEqual(read(root/'progress.json')['remaining'], 1)
            self.assertFalse(read(root/'campaign.json')['complete'])
            self.assertFalse((root/'scoring_complete.json').exists())
            self.assertFalse((root/'results/bad.json').exists())
            self.assertIsNone(read(root/'failures/bad.json')['assigned_score'])


if __name__ == '__main__':
    unittest.main()
