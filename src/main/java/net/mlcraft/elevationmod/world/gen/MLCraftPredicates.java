package net.mlcraft.elevationmod.world.gen;

import net.mlcraft.elevationmod.MLCraft;
import net.mlcraft.elevationmod.accessor.MapInfoAccessor;
import com.mojang.serialization.Codec;
import com.mojang.serialization.codecs.RecordCodecBuilder;
import net.minecraft.registry.Registries;
import net.minecraft.registry.Registry;
import net.minecraft.registry.entry.RegistryEntry;
import net.minecraft.util.dynamic.CodecHolder;
import net.minecraft.world.gen.surfacebuilder.MaterialRules;

public class MLCraftPredicates {
    public static void register() {
        Registry.register(Registries.MATERIAL_CONDITION, MLCraft.id("above_preliminary_surface"), AboveSurfaceMaterialCondition.CODEC.codec());
    }

    /**
     * reimplementation of the above_preliminary_surface rule that reads the atlas.
     * @param depth how far below the surface the rule should extend
     */
    record AboveSurfaceMaterialCondition(int depth) implements MaterialRules.MaterialCondition {
        static final CodecHolder<MLCraftPredicates.AboveSurfaceMaterialCondition> CODEC = CodecHolder.of(
                RecordCodecBuilder.mapCodec(instance -> instance.group(
                                Codec.INT.optionalFieldOf("depth", 5).forGetter(AboveSurfaceMaterialCondition::depth))
                        .apply(instance, MLCraftPredicates.AboveSurfaceMaterialCondition::new)));

        @Override
        public CodecHolder<? extends MaterialRules.MaterialCondition> codec() {
            return CODEC;
        }

        @Override
        public MaterialRules.BooleanSupplier apply(final MaterialRules.MaterialRuleContext materialRuleContext) {
            class AboveSurfacePredicate
                    extends MaterialRules.FullLazyAbstractPredicate {
                AboveSurfacePredicate() {
                    super(materialRuleContext);
                }

                @Override
                protected boolean test() {
                    RegistryEntry<MLCraftMapInfo> amiEntry = ((MapInfoAccessor)(Object) materialRuleContext).mlcraft_getAMI();
                    // should only apply when carvers call this function, which is okay to always have grass
                    if (amiEntry == null) return true;
                    MLCraftMapInfo ami = amiEntry.value();
                    NamespacedMapImage nmi = MLCraft.getOrCreateMap(ami.heightmap(), NamespacedMapImage.Type.GRAYSCALE);
                    double elevation = nmi.getElevation(this.context.blockX, this.context.blockZ, ami.horizontalScale(), ami.horizontalScale(), ami.startingY());
                    return this.context.blockY > elevation - AboveSurfaceMaterialCondition.this.depth;
                }
            }
            return new AboveSurfacePredicate();
        }
    }
}