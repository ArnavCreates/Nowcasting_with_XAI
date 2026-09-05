// Bilinear interpolation over the precomputed 5x5 (wind x precip) risk grid.
// Lets slider positions between grid nodes resolve to a smooth, continuous
// risk score with zero server round-trip.

function findBracket(steps, value) {
  if (value <= steps[0]) return [steps[0], steps[0]];
  if (value >= steps[steps.length - 1]) return [steps[steps.length - 1], steps[steps.length - 1]];

  for (let i = 0; i < steps.length - 1; i++) {
    if (value >= steps[i] && value <= steps[i + 1]) {
      return [steps[i], steps[i + 1]];
    }
  }
  return [steps[0], steps[steps.length - 1]];
}

function getNodeValue(grid, deltaWind, deltaPrecip) {
  const node = grid.find((g) => g.deltaWind === deltaWind && g.deltaPrecip === deltaPrecip);
  return node ? node.riskScore : 0;
}

export function bilinearInterpolate(grid, windSteps, precipSteps, targetWind, targetPrecip) {
  const [w0, w1] = findBracket(windSteps, targetWind);
  const [p0, p1] = findBracket(precipSteps, targetPrecip);

  const q11 = getNodeValue(grid, w0, p0);
  const q21 = getNodeValue(grid, w1, p0);
  const q12 = getNodeValue(grid, w0, p1);
  const q22 = getNodeValue(grid, w1, p1);

  const wRatio = w1 === w0 ? 0 : (targetWind - w0) / (w1 - w0);
  const pRatio = p1 === p0 ? 0 : (targetPrecip - p0) / (p1 - p0);

  const top = q11 + (q21 - q11) * wRatio;
  const bottom = q12 + (q22 - q12) * wRatio;
  const result = top + (bottom - top) * pRatio;

  return Math.round(Math.min(100, Math.max(0, result)) * 10) / 10;
}