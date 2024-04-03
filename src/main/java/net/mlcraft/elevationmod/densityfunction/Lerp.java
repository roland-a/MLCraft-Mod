package net.mlcraft.elevationmod.densityfunction;

import com.mojang.serialization.MapCodec;
import com.mojang.serialization.codecs.RecordCodecBuilder;

import net.minecraft.util.dynamic.CodecHolder;
import net.minecraft.world.gen.densityfunction.DensityFunction;

/**
 * Density function calculating linear interpolation between two values based on the value of the third.
 * <br><br>
 * All three values can be any density function. As opposed to the plain Minecraft version of the operation
 * (see below for code examples), each function gets evaluated at most once, and the "value1" and "value2"
 * functions aren't evaluated if their results wouldn't contribute to the end result due to the "alpha"
 * result being either 1.0 (or above) or 0.0 (or below), respectively.
 * <br><br>
 * The same result can be achieved via a combination of <code>"minecraft:clamp"</code> to limit the "alpha" value
 * to a range 0.0 to 1.0, fed into a chain of <code>"minecraft:add"</code> and <code>"minecraft:mul"</code> codes.
 * Ignoring double evaluation and floating-point arithmetic errors, those three configurations are equivalent:
 * <pre>
 * {
 *   "type": "mlcraft:lerp",
 *   "value1": { ... VALUE 1... },
 *   "value2": { ... VALUe 2 ... },
 *   "alpha": { ... ALPHA ... }
 * }
 *
 * {
 *   "type": "minecraft:add",
 *   "argument1": { ... VALUE 1 ... },
 *   "argument2": {
 *     "type": "minecraft:mul",
 *     "argument1": {
 *       "type": "minecraft:clamp",
 *       "input": { ... ALPHA ... },
 *       "min": 0,
 *       "max": 1
 *     },
 *     "argument2": {
 *       "type": "minecraft:add",
 *       "argument1": { ... VALUE 2 ... },
 *       "argument2": {
 *         "type": "minecraft:mul",
 *         "argument1": { ... VALUE 1 ...},
 *         "argument2": -1
 *       }
 *     }
 *   }
 * }
 *
 * {
 *   "type": "minecraft:add",
 *   "argument1": {
 *     "type": "minecraft:mul",
 *     "argument1": {
 *       "type": "minecraft:add",
 *       "argument1": 1,
 *       "argument2": {
 *         "type": "minecraft:mul",
 *         "argument1": {
 *           "type": "minecraft:clamp",
 *           "input": { ... ALPHA ... },
 *           "min": 0,
 *           "max": 1
 *         },
 *         "argument2": -1
 *       }
 *     },
 *     "argument2": { ... VALUE 1 ... }
 *   },
 *   "argument2": {
 *     "type": "minecraft:mul",
 *     "argument1": {
 *       "type": "minecraft:clamp",
 *       "input": { ... ALPHA ... },
 *       "min": 0,
 *       "max": 1
 *     },
 *     "argument2": { ... VALUE 2 ... }
 *   }
 * }
 * </pre>
 * <strong>ID:</strong><code>"mlcraft:lerp"</code>
 * <br><br>
 * Parameter:
 * <dl>
 * <dt>"value1"</dt><dd>Data type: DensityFunction<br>
 *   The first value, and the result at "alpha" = 0.0 or less</dd>
 * <dt>"value2"</dt><dd>Data type: DensityFunction<br>
 *   The second value, and the result at "alpha" = 1.0 or more</dd>
 * <dt>"alpha"</dt><dd>Data type: DensityFunction<br>
 *   The first value, and the result at "alpha" = 0.0</dd>
 * <dt>Return value</dt><dd>When "alpha" returns 0.0 or less, the result of the "value1" function<br>
 *   When "alpha" returns 1.0 or more, the result of the "value2" function<br>
 *   A linear interpolation between those two with "alpha" providing the weight otherwise</dd>
 * </dl>
 * <br><br>
 * See also: <code>"minecraft:spline"</code> for a flexible way to pre-process the "alpha" function,
 * creating near arbitrary blending profiles.
 */
public record Lerp(DensityFunction value1, DensityFunction value2, DensityFunction alpha) implements DensityFunction {
    public static final MapCodec<Lerp> LERP_CODEC = RecordCodecBuilder.mapCodec(
            (instance) -> instance.group(
                    (DensityFunction.FUNCTION_CODEC.fieldOf("value1")).forGetter(Lerp::value1),
                    (DensityFunction.FUNCTION_CODEC.fieldOf("value2")).forGetter(Lerp::value2),
                    (DensityFunction.FUNCTION_CODEC.fieldOf("alpha")).forGetter(Lerp::alpha)).apply(instance, Lerp::new));
    private static final CodecHolder<? extends DensityFunction> CODEC_HOLDER = CodecHolder.of(LERP_CODEC);

    @Override public double sample(NoisePos pos) {
        double alphaVal = alpha.sample(pos);
        if(alphaVal <= 0.0) {
            return value1.sample(pos);
        }
        if(alphaVal >= 1.0) {
            return value2.sample(pos);
        }
        return (1 - alphaVal) * value1.sample(pos) + alphaVal * value2.sample(pos);
    }

    @Override public void fill(double[] densities, EachApplier applier) {
        double[] alphas = new double[densities.length];
        alpha.fill(alphas, applier);
        for (int i = 0; i < densities.length; ++i) {
            if(alphas[i] <= 0.0) {
                densities[i] = value1.sample(applier.at(i));
            } else if(alphas[i] >= 1.0) {
                densities[i] = value2.sample(applier.at(i));
            } else {
                densities[i] = (1 - alphas[i]) * value1.sample(applier.at(i)) + alphas[i] * value2.sample(applier.at(i));
            }
        }
    }

    @Override public DensityFunction apply(DensityFunctionVisitor visitor) {
        return new Lerp(value1.apply(visitor), value2.apply(visitor), alpha.apply(visitor));
    }

    @Override public double minValue() {
        return Math.min(value1.minValue(), value2.minValue());
    }

    @Override public double maxValue() {
        return Math.max(value1.maxValue(), value2.maxValue());
    }

    @Override public CodecHolder<? extends DensityFunction> getCodecHolder() {
        return CODEC_HOLDER;
    }

}

