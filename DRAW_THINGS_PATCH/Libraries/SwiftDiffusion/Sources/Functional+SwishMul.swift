import NNC

extension Functional {
  public static func swishMul(
    value: Model.IO,
    gate: Model.IO,
    beta: Float = 1,
    scale: Float = 1,
    streamContext: StreamContext? = nil
  ) -> Model.IO {
    let activated = gate.swish(beta: beta)
    if scale == 1 {
      return value .* activated
    }
    return scale * (value .* activated)
  }

  public static func swishMul<T: DynamicGraph.TensorGroup>(
    value: T,
    gate: T,
    beta: Float = 1,
    scale: Float = 1,
    streamContext: StreamContext? = nil
  ) -> T {
    let activated = Functional.swish(gate, beta: beta, streamContext: streamContext)
    if scale == 1 {
      return value .* activated
    }
    return scale * (value .* activated)
  }
}