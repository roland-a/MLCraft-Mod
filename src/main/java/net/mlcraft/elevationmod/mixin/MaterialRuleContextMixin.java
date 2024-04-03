package net.mlcraft.elevationmod.mixin;

import net.mlcraft.elevationmod.accessor.MapInfoAccessor;
import net.mlcraft.elevationmod.world.gen.MLCraftMapInfo;
import net.minecraft.registry.entry.RegistryEntry;
import net.minecraft.world.gen.surfacebuilder.MaterialRules;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;

@Mixin(MaterialRules.MaterialRuleContext.class)
public class MaterialRuleContextMixin implements MapInfoAccessor {
    @Unique
    private RegistryEntry<MLCraftMapInfo> mlcraft_AMI;

    @Override
    public RegistryEntry<MLCraftMapInfo> mlcraft_getAMI() {
        return this.mlcraft_AMI;
    }

    @Override
    public void mlcraft_setAMI(RegistryEntry<MLCraftMapInfo> ami) {
        this.mlcraft_AMI = ami;
    }


}