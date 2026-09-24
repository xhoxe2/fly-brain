import assert from "node:assert/strict";
import test from "node:test";
import { TasteIntroduction, TASTE_STEPS } from "./introduction.ts";

test("taste introduction waits for live computation and retains a continuous model clock", () => {
  const tour = new TasteIntroduction(25_000);
  assert.equal(tour.advance(8000, 25_500), false, "do not outrun a slow brain");
  assert.equal(tour.progress(25_500), 0.5);
  assert.equal(tour.advance(0, 25_500), false, "a paused frame cannot advance");
  assert.equal(tour.advance(100, 26_000), true);
  assert.equal(TASTE_STEPS[tour.step].bitter, 1);
  assert.equal(tour.advance(7999, 30_000), false, "fast computation cannot rush a step");
  assert.equal(tour.advance(1, 30_000), true);
  assert.equal(TASTE_STEPS[tour.step].bitter, 0);
  assert.equal(tour.advance(8000, 34_000), true);
  assert.equal(tour.done, true);
  assert.equal(tour.advance(8000, 38_000), false, "do not loop or alter conditions after finishing");
});
